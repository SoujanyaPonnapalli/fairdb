//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "rocksdb/write_buffer_manager.h"

#include <memory>
#include "cache/cache_entry_roles.h"
#include "cache/cache_reservation_manager.h"
#include "db/db_impl/db_impl.h"
#include "rocksdb/status.h"
#include "util/coding.h"
#include "rocksdb/tg_thread_local.h"

#include <execinfo.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include <vector>
#include <mutex>

namespace ROCKSDB_NAMESPACE {

WriteBufferManager::WriteBufferManager(size_t buffer_size,
                                       std::shared_ptr<Cache> cache,
                                       bool allow_stall,
                                       size_t num_clients,
                                       size_t steady_res_size)
    : buffer_size_(buffer_size),
      mutable_limit_(buffer_size),
      global_size_(buffer_size - steady_res_size),  // Initialize global pool size to buffer size
      global_used_(0),
      per_client_global_used_(num_clients),
      steady_res_size_(steady_res_size),
      per_client_is_steady_(num_clients, false),
      steady_res_used_(0),
      per_client_steady_res_used_(num_clients),
      per_client_buffer_size_(num_clients, buffer_size),
      memory_active_(0),
      cache_res_mgr_(nullptr),
      per_client_queue_(num_clients),
      allow_stall_(allow_stall),
      per_client_stall_active_(num_clients),
      per_client_stall_count_(num_clients),
      total_stall_count_(0) {
  std::cout << "[FAIRDB_LOG] WBM Size: " << (buffer_size / 1024 / 1024) << " MB" << std::endl;
  std::cout << "[FAIRDB_LOG] Global Pool Size: " << (global_size_ / 1024 / 1024) << " MB" << std::endl;
  std::cout << "[FAIRDB_LOG] Steady Res Size: " << (steady_res_size_ / 1024 / 1024) << " MB" << std::endl;
  for (size_t i = 0; i < num_clients; ++i) {
    per_client_global_used_[i] = 0;
    per_client_steady_res_used_[i] = 0;
    per_client_stall_active_[i] = false;
    per_client_stall_count_[i] = 0;
  }

  if (cache) {
    // Memtable's memory usage tends to fluctuate frequently
    // therefore we set delayed_decrease = true to save some dummy entry
    // insertion on memory increase right after memory decrease
    cache_res_mgr_ = std::make_shared<
        CacheReservationManagerImpl<CacheEntryRole::kWriteBuffer>>(
        cache, true /* delayed_decrease */);
  }

  mt_log_file_.open("logs/memtable_stats.txt", std::ios::out | std::ios::trunc);
  // Ensure log file is open and ready for use
  if (!mt_log_file_.is_open()) {
    throw std::ios_base::failure("Failed to open log/memtable_stats.txt");
  }
}

WriteBufferManager::~WriteBufferManager() {
#ifndef NDEBUG
  std::unique_lock<std::mutex> lock(mu_);
  for (const auto& queue : per_client_queue_) {
    assert(queue.empty());
  }
#endif
  if (mt_log_file_.is_open()) {
    mt_log_file_.close();
  }
}

std::size_t WriteBufferManager::dummy_entries_in_cache_usage() const {
  if (cache_res_mgr_ != nullptr) {
    return cache_res_mgr_->GetTotalReservedCacheSize();
  } else {
    return 0;
  }
}

bool WriteBufferManager::ShouldStall(int client_id) const {
  if (client_id < 0) {
    std::cout << "[FAIRDB_LOG] Unaccounted ShouldStall " << client_id << std::endl;
    return false;
  }
  if (!allow_stall_.load(std::memory_order_relaxed) || !enabled()) {
    return false;
  }

  return IsStallActive(client_id) || IsStallThresholdExceeded(client_id);
}

bool WriteBufferManager::IsStallThresholdExceeded(int client_id) const {
  size_t client_total_usage = per_client_memory_usage(client_id);

  // Stall if client's total usage exceeds their buffer size
  if (client_total_usage >= per_client_buffer_size_[client_id]) {
    if (per_client_is_steady_[client_id]) {
      std::cout << "[FAIRDB_LOG] Stalling steady client " << client_id << ". Total usage=" << client_total_usage << std::endl;
    } else {
      std::cout << "[FAIRDB_LOG] Stalling NON-steady client " << client_id << ". Total usage=" << client_total_usage << std::endl;
    }
    return true;
  }

  if (per_client_is_steady_[client_id]) {
    // Steady client: Stall if both global and steady pools are at or above capacity
    size_t global_used = global_used_.load(std::memory_order_relaxed);
    bool global_full = global_used >= global_size_;
    size_t steady_used = steady_res_used_.load(std::memory_order_relaxed);
    bool steady_full = steady_used >= steady_res_size_;
    if (global_full && steady_full) {
      std::cout << "[FAIRDB_LOG] Stalling steady client " << client_id << ". Global=" << global_used << ", Steady=" << steady_used << std::endl;
    }
    return global_full && steady_full;
  } else {
    // Non-steady client: Stall if global pool is at or above capacity
    bool should_stall = global_used_.load(std::memory_order_relaxed) >= global_size_;
    if (should_stall) {
      std::cout << "[FAIRDB_LOG] Stalling NON-steady client " << client_id << ". Global usage=" << global_used_.load(std::memory_order_relaxed) << std::endl;
    }
    return should_stall;
  }
}


void WriteBufferManager::SetPerClientBufferSize(int client_id, size_t buffer_size) {
  per_client_buffer_size_[client_id] = buffer_size;
}

void WriteBufferManager::SetSteadyReservationSize(size_t steady_size) {
  steady_res_size_ = steady_size;
  global_size_ = buffer_size_.load(std::memory_order_relaxed) - steady_size;
  assert(global_size_ >= 0);  // Ensure global_size_ is not negative
}

void WriteBufferManager::SetClientAsSteady(int client_id, bool is_steady) {
  per_client_is_steady_[client_id] = is_steady;
}

size_t WriteBufferManager::per_client_global_usage(int client_id) const {
  return per_client_global_used_[client_id].load(std::memory_order_relaxed);
}

size_t WriteBufferManager::per_client_steady_usage(int client_id) const {
  return per_client_steady_res_used_[client_id].load(std::memory_order_relaxed);
}

size_t WriteBufferManager::aggregate_global_usage() const {
  return global_used_.load(std::memory_order_relaxed);
}

size_t WriteBufferManager::aggregate_steady_usage() const {
  return steady_res_used_.load(std::memory_order_relaxed);
}

void WriteBufferManager::ReserveMem(size_t mem) {
  int client_id = TG_GetThreadMetadata().client_id;
  if (client_id < 0) {
    std::cout << "[FAIRDB_LOG] Unaccounted ReserveMem " << client_id << std::endl;
    return;
  }
  bool reserved_from_global = false;
  if (cache_res_mgr_ != nullptr) {
    ReserveMemWithCache(mem);
  } else if (enabled()) {
    if (per_client_is_steady_[client_id]) {
      // Steady client
      size_t steady_left = steady_res_size_ - steady_res_used_.load(std::memory_order_relaxed);
      if (steady_left >= mem) {
        // Allocate from steady pool
        per_client_steady_res_used_[client_id].fetch_add(mem, std::memory_order_relaxed);
        steady_res_used_.fetch_add(mem, std::memory_order_relaxed);
      } else {
        // Allocate from global pool
        per_client_global_used_[client_id].fetch_add(mem, std::memory_order_relaxed);
        global_used_.fetch_add(mem, std::memory_order_relaxed);
        reserved_from_global = true;
      }
    } else {
      // Non-steady client
      per_client_global_used_[client_id].fetch_add(mem, std::memory_order_relaxed);
      global_used_.fetch_add(mem, std::memory_order_relaxed);
      reserved_from_global = true;
    }
  }
  if (enabled()) {
    memory_active_.fetch_add(mem, std::memory_order_relaxed);
  }
  mt_log_file_ << "wbm," << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count()
               << "," << client_id << ",res," << mem << ","
               << per_client_memory_usage(client_id) << "," << (reserved_from_global ? "global" : "steady") << std::endl;
}

void WriteBufferManager::ScheduleFreeMem(size_t mem) {
  if (enabled()) {
    memory_active_.fetch_sub(mem, std::memory_order_relaxed);
  }
}

void WriteBufferManager::FreeMem(size_t mem) {
  int client_id = TG_GetThreadMetadata().client_id;
  if (client_id < 0) {
    std::cout << "[FAIRDB_LOG] Unaccounted FreeMem " << client_id << std::endl;
    return;
  }
  size_t freed_from_global = 0;
  size_t freed_from_steady = 0;
  if (cache_res_mgr_ != nullptr) {
    FreeMemWithCache(mem);
  } else if (enabled()) {
    if (per_client_is_steady_[client_id]) {
      // Steady client
      size_t client_global_usage = per_client_global_used_[client_id].load(std::memory_order_relaxed);
      size_t free_from_global = std::min(mem, client_global_usage);
      freed_from_global = free_from_global;

      // Avoid underflow when freeing global memory
      per_client_global_used_[client_id].store(
          client_global_usage >= free_from_global ? 
          client_global_usage - free_from_global : 0, 
          std::memory_order_relaxed);
      global_used_.store(
          global_used_.load(std::memory_order_relaxed) >= free_from_global ? 
          global_used_.load(std::memory_order_relaxed) - free_from_global : 0, 
          std::memory_order_relaxed);

      size_t remaining_mem = mem - free_from_global;
      freed_from_steady = remaining_mem;

      if (remaining_mem > 0) {
        // Free from steady reservation, avoiding underflow
        size_t client_steady_usage = per_client_steady_res_used_[client_id].load(std::memory_order_relaxed);
        per_client_steady_res_used_[client_id].store(
            client_steady_usage >= remaining_mem ? 
            client_steady_usage - remaining_mem : 0, 
            std::memory_order_relaxed);
        steady_res_used_.store(
            steady_res_used_.load(std::memory_order_relaxed) >= remaining_mem ? 
            steady_res_used_.load(std::memory_order_relaxed) - remaining_mem : 0, 
            std::memory_order_relaxed);
      }
    } else {
      // Non-steady client
      size_t client_global_usage = per_client_global_used_[client_id].load(std::memory_order_relaxed);
      per_client_global_used_[client_id].store(
          client_global_usage >= mem ? 
          client_global_usage - mem : 0, 
          std::memory_order_relaxed);
      global_used_.store(
          global_used_.load(std::memory_order_relaxed) >= mem ? 
          global_used_.load(std::memory_order_relaxed) - mem : 0, 
          std::memory_order_relaxed);
      freed_from_global = mem;
    }
  }
  mt_log_file_ << "wbm," << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count()
               << "," << client_id << ",free," << mem << ","
               << per_client_memory_usage(client_id) << ",global:" << freed_from_global << ",steady:" << freed_from_steady << std::endl;

  // Check if stall is active and can be ended.
  MaybeEndWriteStall();
}

void WriteBufferManager::ReserveMemWithCache(size_t mem) {
  assert(cache_res_mgr_ != nullptr);
  std::cout << "[FAIRDB_LOG] Surprise! ReserveMemWithCache was called.\n";
  // Use a mutex to protect various data structures. Can be optimized to a
  // lock-free solution if it ends up with a performance bottleneck.
  std::lock_guard<std::mutex> lock(cache_res_mgr_mu_);

  size_t new_mem_used = memory_usage() + mem;
  Status s = cache_res_mgr_->UpdateCacheReservation(new_mem_used);

  // We absorb the error since WriteBufferManager is not able to handle
  // this failure properly. Ideally we should prevent this allocation
  // from happening if this cache charging fails.
  s.PermitUncheckedError();
}

void WriteBufferManager::FreeMemWithCache(size_t mem) {
  assert(cache_res_mgr_ != nullptr);
  std::cout << "[FAIRDB_LOG] Surprise! FreeMemWithCache was called.\n";

  std::lock_guard<std::mutex> lock(cache_res_mgr_mu_);

  size_t new_mem_used = memory_usage() >= mem ? memory_usage() - mem : 0;

  Status s = cache_res_mgr_->UpdateCacheReservation(new_mem_used);

  // We absorb the error since WriteBufferManager is not able to handle
  // this failure properly.
  s.PermitUncheckedError();
}

void WriteBufferManager::BeginWriteStall(StallInterface* wbm_stall, int client_id) {
  assert(wbm_stall != nullptr);
  if (client_id < 0) {
    std::cout << "[FAIRDB_LOG] Unaccounted BeginWriteStall " << client_id << std::endl;
    return;
  }
  // Increment the stall count for the client atomically
  per_client_stall_count_[client_id].fetch_add(1, std::memory_order_relaxed);
  size_t current_total_stalls = total_stall_count_.fetch_add(1, std::memory_order_relaxed) + 1;

  // Print stall counts every 1000 total stalls
  if (current_total_stalls % 1000 == 0) {
    std::cout << "Stall counts:" << std::endl;
    for (size_t i = 0; i < per_client_stall_count_.size(); ++i) {
      std::cout << "Client " << i << ": " << per_client_stall_count_[i].load(std::memory_order_relaxed) << " stalls" << std::endl;
    }
  }

  std::list<StallInterface*> new_node = {wbm_stall};
  {
    std::unique_lock<std::mutex> lock(mu_);
    // Verify if the stall conditions are still active.
    if (ShouldStall(client_id)) {
      per_client_stall_active_[client_id].store(true, std::memory_order_relaxed);
      per_client_queue_[client_id].splice(per_client_queue_[client_id].end(), std::move(new_node));
    }
  }

  // If the node was not consumed, the stall has ended already and we can signal
  // the caller.
  if (!new_node.empty()) {
    new_node.front()->Signal();
  }
}

// Called when memory is freed in FreeMem or the buffer size has changed.
void WriteBufferManager::MaybeEndWriteStall() {
  // Perform all deallocations outside of the lock.
  std::vector<std::list<StallInterface*>> cleanup(per_client_queue_.size());

  std::unique_lock<std::mutex> lock(mu_);
  for (size_t client_id = 0; client_id < per_client_queue_.size(); ++client_id) {
    if (!per_client_stall_active_[client_id].load(std::memory_order_relaxed)) {
      continue;  // Nothing to do for this client.
    }

    // Stall conditions have not been resolved for this client.
    if (allow_stall_.load(std::memory_order_relaxed) &&
        IsStallThresholdExceeded(client_id)) {
      continue;
    }

    // Unblock new writers for this client.
    per_client_stall_active_[client_id].store(false, std::memory_order_relaxed);

    // Unblock the writers in the queue.
    for (StallInterface* wbm_stall : per_client_queue_[client_id]) {
      wbm_stall->Signal();
    }
    cleanup[client_id] = std::move(per_client_queue_[client_id]);
  }
}

void WriteBufferManager::RemoveDBFromQueue(StallInterface* wbm_stall, int client_id) {
  assert(wbm_stall != nullptr);

  // Deallocate the removed nodes outside of the lock.
  std::list<StallInterface*> cleanup;

  if (enabled() && allow_stall_.load(std::memory_order_relaxed)) {
    std::unique_lock<std::mutex> lock(mu_);
    for (auto it = per_client_queue_[client_id].begin(); it != per_client_queue_[client_id].end();) {
      auto next = std::next(it);
      if (*it == wbm_stall) {
        cleanup.splice(cleanup.end(), per_client_queue_[client_id], std::move(it));
      }
      it = next;
    }
  }
  wbm_stall->Signal();
}

}  // namespace ROCKSDB_NAMESPACE
