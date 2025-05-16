//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// WriteBufferManager is for managing memory allocation for one or more
// MemTables.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <list>
#include <mutex>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "rocksdb/cache.h"
#include "rocksdb/status.h"
#include "rocksdb/tg_thread_local.h"

namespace ROCKSDB_NAMESPACE {

class CacheReservationManager;

// Interface to block and signal DB instances, intended for RocksDB
// internal use only. Each DB instance contains ptr to StallInterface.
class StallInterface {
 public:
  virtual ~StallInterface() {}

  virtual void Block() = 0;

  virtual void Signal() = 0;
};

class WriteBufferManager final {
 private:
  // Manages WAL size limits based on column family reservations.
  // Allows internal tracking to go below zero, representing overshoot.
  // External checks enforce the actual limits.
  class WALManager {
   public:
    WALManager()
        : max_total_wal_size_(0),
          total_reserved_size_(0),
          global_pool_remaining_(0),
          enabled_(false) {}

    // Sets the maximum total size the WALs can reach.
    // This should be called before setting any reservations.
    // REQUIRES: mu_ held.
    void SetMaxTotalWALSize(size_t size, size_t current_total_wal_size) {
      max_total_wal_size_ = size;
      // Initialize remaining global pool based on current usage and reservations
      RecalculateGlobalPoolRemaining(current_total_wal_size);
      CheckEnabled();
    }

    // Sets the reserved WAL space for a specific column family.
    // REQUIRES: mu_ held.
    void SetColumnFamilyReservation(uint32_t cf_id, size_t size,
                                      size_t current_total_wal_size) {
      size_t old_reservation = cf_total_reservation_[cf_id]; // Default 0 if not exists
      size_t reservation_diff = size - old_reservation;

      cf_total_reservation_[cf_id] = size;
      total_reserved_size_ += reservation_diff;

      // Initialize/update usage maps if CF is new
       if (cf_reserved_usage_.find(cf_id) == cf_reserved_usage_.end()) {
           cf_reserved_usage_[cf_id] = 0;
       }
       if (cf_global_usage_.find(cf_id) == cf_global_usage_.end()) {
           cf_global_usage_[cf_id] = 0;
       }

      // Adjust usage if reservation decreased below current reserved usage
      if (size < cf_reserved_usage_[cf_id]) {
          size_t excess_usage = cf_reserved_usage_[cf_id] - size;
          cf_reserved_usage_[cf_id] = size;
          // This excess usage logically moves to the global pool
          cf_global_usage_[cf_id] += excess_usage;
          // Global pool remaining decreases accordingly
          // Note: RecalculateGlobalPoolRemaining handles the overall adjustment
      }

      if (max_total_wal_size_ > 0) {
        RecalculateGlobalPoolRemaining(current_total_wal_size);
      }
      CheckEnabled();
    }

    // Checks if a column family can write the specified size to the WAL based
    // on current limits.
    // Returns OK if space is available according to rules.
    // Returns Status::MemoryLimit if a flush is needed for this CF.
    // REQUIRES: mu_ held.
    Status CheckCanWriteWAL(uint32_t cf_id, size_t write_size) const {
      if (!enabled_) {
        return Status::OK();
      }

      size_t current_reserved_usage = 0;
      auto reserved_it = cf_reserved_usage_.find(cf_id);
      if (reserved_it != cf_reserved_usage_.end()) {
          current_reserved_usage = reserved_it->second;
      }

      size_t reservation = 0;
      auto total_res_it = cf_total_reservation_.find(cf_id);
      if (total_res_it != cf_total_reservation_.end()) {
          reservation = total_res_it->second;
      }

      // Available reserved space cannot be negative for the check
      size_t available_reserved = (reservation > current_reserved_usage) ? (reservation - current_reserved_usage) : 0;
      // Available global space for the check cannot be negative either
      size_t available_global = (global_pool_remaining_ > 0) ? static_cast<size_t>(global_pool_remaining_) : 0;


      if (write_size <= available_reserved + available_global) {
        return Status::OK();
      } else {
        // Not enough space even combining reserved and global
        return Status::MemoryLimit(
            "WAL space limit reached for CF, flush required");
      }
    }

    // Attributes WAL space usage to a column family. Internal accounting allows
    // global_pool_remaining_ to become negative.
    // REQUIRES: mu_ held.
    void AttributeWALSpace(uint32_t cf_id, size_t size) {
      if (!enabled_ || size == 0) {
        return;
      }

      size_t reservation = 0;
      auto total_res_it = cf_total_reservation_.find(cf_id);
       if (total_res_it != cf_total_reservation_.end()) {
           reservation = total_res_it->second;
       }

      size_t current_reserved_usage = 0;
      auto reserved_it = cf_reserved_usage_.find(cf_id);
       if (reserved_it != cf_reserved_usage_.end()) {
           current_reserved_usage = reserved_it->second;
       }

      size_t use_from_reserved = std::min(size, reservation - current_reserved_usage);
      // Ensure map entry exists before incrementing
      cf_reserved_usage_[cf_id] = current_reserved_usage + use_from_reserved;


      size_t use_from_global = size - use_from_reserved;
      if (use_from_global > 0) {
        // No assertion here, allow overshoot internally
        global_pool_remaining_ -= use_from_global;
        // Ensure map entry exists before incrementing
        cf_global_usage_[cf_id] = cf_global_usage_[cf_id] + use_from_global;
      }
    }

    // Deattributes WAL space usage from a column family (e.g., after flush).
    // REQUIRES: mu_ held.
    void DeattributeWALSpace(uint32_t cf_id, size_t size) {
      if (!enabled_ || size == 0) {
        return;
      }

       size_t current_global_usage = 0;
       auto global_it = cf_global_usage_.find(cf_id);
       if (global_it != cf_global_usage_.end()) {
           current_global_usage = global_it->second;
       }

       size_t deduct_from_global = std::min(size, current_global_usage);
       if (deduct_from_global > 0) {
           cf_global_usage_[cf_id] -= deduct_from_global; // Decrement existing entry
           global_pool_remaining_ += deduct_from_global;
       }

       size_t deduct_from_reserved = size - deduct_from_global;
       if (deduct_from_reserved > 0) {
            auto reserved_it = cf_reserved_usage_.find(cf_id);
            // If we are deducting from reserved, it must exist and have enough usage
            assert(reserved_it != cf_reserved_usage_.end());
            assert(reserved_it->second >= deduct_from_reserved);
            cf_reserved_usage_[cf_id] -= deduct_from_reserved; // Decrement existing entry
       }
    }

    // Checks if reserved space is available for flushing CFs after WAL
    // reattribution.
    // Returns the number of bytes of global space usage that exceed the
    // allowed limit (0 if within limits or disabled). Flushing oldest CFs might
    // be needed if the return value is > 0.
    // REQUIRES: mu_ held.
    size_t CheckEnsureReservedSpaceAvailable() const {
        if (!enabled_) {
            return 0; // No limit enforced
        }

        size_t current_total_global_usage = GetCurrentTotalGlobalUsage();
        size_t max_allowed_global_usage = GetMaxAllowedGlobalUsage();

        if (current_total_global_usage > max_allowed_global_usage) {
            return current_total_global_usage - max_allowed_global_usage;
        }
        return 0; // Within limits
    }

    bool IsEnabled() const { return enabled_; }

    // Logs the current usage statistics to the provided logger.
    // REQUIRES: mu_ held.
    std::string LogUsage() const {
       // Removed check and ROCKS_LOG_INFO call here - logging is done by caller

       std::stringstream ss;
       if (!enabled_) {
           ss << "WALManager: Disabled (no reservations or max size set)";
           return ss.str();
       }

       ss << "WALManager Usage:" << std::endl;
       ss << "  Max Total WAL Size: " << max_total_wal_size_ << " bytes" << std::endl;
       ss << "  Total Reserved Size: " << total_reserved_size_ << " bytes" << std::endl;
       // global_pool_remaining_ can be negative here
       ss << "  Internal Global Pool Remaining: " << global_pool_remaining_ << " bytes" << std::endl;
       ss << "  Max Allowed Global Usage: " << GetMaxAllowedGlobalUsage() << " bytes" << std::endl;
       ss << "  Current Total Global Usage: " << GetCurrentTotalGlobalUsage() << " bytes" << std::endl;
       ss << "  Per-CF Usage (Reservation | Reserved Used | Global Used):" << std::endl;

       // Iterate over reservations as the primary source of known CFs
       for (const auto& res_pair : cf_total_reservation_) {
           uint32_t cf_id = res_pair.first;
           size_t reservation = res_pair.second;

           size_t reserved_used = 0;
           auto res_used_it = cf_reserved_usage_.find(cf_id);
           if (res_used_it != cf_reserved_usage_.end()) {
               reserved_used = res_used_it->second;
           }

           size_t global_used = 0;
           auto global_used_it = cf_global_usage_.find(cf_id);
           if (global_used_it != cf_global_usage_.end()) {
               global_used = global_used_it->second;
           }

           ss << "    CF[" << cf_id << "]: " << reservation << " | "
              << reserved_used << " | " << global_used << std::endl;
       }
       // Log CFs that might only have global usage
       for (const auto& global_pair : cf_global_usage_) {
           uint32_t cf_id = global_pair.first;
           if (cf_total_reservation_.find(cf_id) == cf_total_reservation_.end() && global_pair.second > 0) {
                 ss << "    CF[" << cf_id << "] (No Res): 0 | 0 | " << global_pair.second << std::endl;
           }
       }
       return ss.str();
    }

   private:
    // Recalculates the remaining global pool size based on total size,
    // reservations, and current usage. Can result in a negative value.
    // REQUIRES: mu_ held.
    void RecalculateGlobalPoolRemaining(size_t current_total_wal_size) {
        if (max_total_wal_size_ == 0) {
            global_pool_remaining_ = 0;
            return;
        }
        // Size calculation is based on current reported size, reservation, and usage.
        // However, the internal remaining tracker is simpler.
        size_t current_total_global_usage = GetCurrentTotalGlobalUsage();
        size_t max_global_pool_size = GetMaxAllowedGlobalUsage();


        // Remaining is simply the max possible minus what's currently used globally.
        // This allows global_pool_remaining_ to become negative.
        global_pool_remaining_ = static_cast<int64_t>(max_global_pool_size) - static_cast<int64_t>(current_total_global_usage);

        // Consistency check (optional, for debugging):
        // size_t current_total_reserved_usage = 0;
        // for(const auto& pair : cf_reserved_usage_) {
        //     current_total_reserved_usage += pair.second;
        // }
        // assert(static_cast<int64_t>(current_total_reserved_usage) + static_cast<int64_t>(current_total_global_usage) <= static_cast<int64_t>(current_total_wal_size) || global_pool_remaining_ < 0);
    }

    // Checks if the manager should be enabled based on reservations.
    // REQUIRES: mu_ held.
    void CheckEnabled() {
      // Enable only if max size is set and there's at least one reservation.
      enabled_ = (max_total_wal_size_ > 0 && total_reserved_size_ > 0);
    }

    // Calculates the total space currently used from the global pool by all CFs.
    // REQUIRES: mu_ held.
    size_t GetCurrentTotalGlobalUsage() const {
        size_t total = 0;
        for (const auto& pair : cf_global_usage_) {
            total += pair.second;
        }
        return total;
    }

    // Calculates the maximum theoretical size of the global pool.
    // REQUIRES: mu_ held.
     size_t GetMaxAllowedGlobalUsage() const {
         if (max_total_wal_size_ > total_reserved_size_) {
             return max_total_wal_size_ - total_reserved_size_;
         }
         return 0;
     }


    size_t max_total_wal_size_;
    size_t total_reserved_size_; // Sum of all cf_total_reservation_ values
    // Can be negative to represent overshoot
    int64_t global_pool_remaining_;
    bool enabled_;

    // Map<cf_id, reservation_size>
    std::map<uint32_t, size_t> cf_total_reservation_;
    // Map<cf_id, reserved_space_currently_used>
    std::map<uint32_t, size_t> cf_reserved_usage_;
     // Map<cf_id, global_space_currently_used>
    std::map<uint32_t, size_t> cf_global_usage_;

  }; // End WALManager class

 public:
  // Parameters:
  // buffer_size: buffer_size = 0 indicates no limit. Memory won't be capped.
  // memory_usage() won't be valid and ShouldFlush() will always return false.
  //
  // cache: if `cache` is provided, we'll put dummy entries in the cache and
  // cost the memory allocated to the cache. It can be used even if buffer_size
  // = 0.
  //
  // allow_stall: if set true, it will enable stalling of writes when
  // memory_usage() exceeds buffer_size. It will wait for flush to complete and
  // memory usage to drop down.
  explicit WriteBufferManager(size_t buffer_size,
                              std::shared_ptr<Cache> cache = {},
                              bool allow_stall = true,
                              size_t num_clients = 16, 
                              size_t steady_res_size = 0);

  // No copying allowed
  WriteBufferManager(const WriteBufferManager&) = delete;
  WriteBufferManager& operator=(const WriteBufferManager&) = delete;

  ~WriteBufferManager();

  // Returns true if buffer_limit is passed to limit the total memory usage and
  // is greater than 0.
  bool enabled() const { return buffer_size() > 0; }

  // Returns true if pointer to cache is passed.
  bool cost_to_cache() const { return cache_res_mgr_ != nullptr; }

  // Returns the total memory used by memtables.
  // Only valid if enabled()
  size_t memory_usage() const {
    return global_used_.load(std::memory_order_relaxed) +
           steady_res_used_.load(std::memory_order_relaxed);
  }

  // Returns the per-client memory usage.
  size_t per_client_memory_usage(int client_id) const {
    return per_client_global_used_[client_id].load(std::memory_order_relaxed) +
           per_client_steady_res_used_[client_id].load(std::memory_order_relaxed);
  }

  // Returns the total memory used by active memtables.
  size_t mutable_memtable_memory_usage() const {
    return memory_active_.load(std::memory_order_relaxed);
  }

  size_t dummy_entries_in_cache_usage() const;

  // Returns the buffer_size.
  size_t buffer_size() const {
    return buffer_size_.load(std::memory_order_relaxed);
  }

  size_t num_clients() const {
    return per_client_global_used_.size();
  }

  // Sets the per-client buffer size (memory usage threshold) for stalling.
  void SetPerClientBufferSize(int client_id, size_t buffer_size);

  // Set the steady reservation size and adjust global pool size accordingly.
  void SetSteadyReservationSize(size_t steady_size);

  // Mark a client as steady or non-steady.
  void SetClientAsSteady(int client_id, bool is_steady);

  // Get per-client global pool usage.
  size_t per_client_global_usage(int client_id) const;

  // Get per-client steady pool usage.
  size_t per_client_steady_usage(int client_id) const;

  // Get aggregate global pool usage.
  size_t aggregate_global_usage() const;

  // Get aggregate steady pool usage.
  size_t aggregate_steady_usage() const;

  // REQUIRED: `new_size` > 0
  void SetBufferSize(size_t new_size) {
    assert(new_size > 0);
    buffer_size_.store(new_size, std::memory_order_relaxed);
    mutable_limit_.store(new_size * 7 / 8, std::memory_order_relaxed);
    // Adjust global and steady pool sizes
    global_size_ = buffer_size_ - steady_res_size_;
    // Check if stall is active and can be ended.
    MaybeEndWriteStall();
  }

  void SetAllowStall(bool new_allow_stall) {
    allow_stall_.store(new_allow_stall, std::memory_order_relaxed);
    MaybeEndWriteStall();
  }

  // Below functions should be called by RocksDB internally.

  // Should only be called from write thread
  bool ShouldFlush() const {
    if (enabled()) {
      if (mutable_memtable_memory_usage() >
          mutable_limit_.load(std::memory_order_relaxed)) {
        return true;
      }
      size_t local_size = buffer_size();
      if (memory_usage() >= local_size &&
          mutable_memtable_memory_usage() >= local_size / 2) {
      // If the memory exceeds the buffer size, we trigger more aggressive
      // flush. But if already more than half memory is being flushed,
      // triggering more flush may not help. We will hold it instead.
        return true;
      }
    }
    return false;
  }

  // Returns true if total memory usage exceeded buffer_size.
  // We stall the writes until memory_usage drops below buffer_size. When the
  // function returns true, the writer thread for the given client will be stalled.
  // Stall is allowed only if user passes allow_stall = true during
  // WriteBufferManager instance creation.
  //
  // Should only be called by RocksDB internally.
  bool ShouldStall(int client_id) const;

  // Returns true if stall is active for the given client.
  bool IsStallActive(int client_id) const {
    return per_client_stall_active_[client_id].load(std::memory_order_relaxed);
  }

  // Returns true if stalling condition is met for the given client.
  bool IsStallThresholdExceeded(int client_id) const;

  void ReserveMem(size_t mem);

  // We are in the process of freeing `mem` bytes, so it is not considered
  // when checking the soft limit.
  void ScheduleFreeMem(size_t mem);

  void FreeMem(size_t mem);

  // Add the DB instance to the per-client queue and block the DB.
  // Should only be called by RocksDB internally.
  void BeginWriteStall(StallInterface* wbm_stall, int client_id);

  // If stall conditions have resolved, remove DB instances from queue and
  // signal them to continue.
  void MaybeEndWriteStall();

  void RemoveDBFromQueue(StallInterface* wbm_stall, int client_id);

  // Sets the overall maximum size for the WAL files. This implicitly defines
  // the global pool size after accounting for reservations.
  // Needs the current total size of WAL files to initialize correctly.
  void SetMaxTotalWALSize(size_t size, size_t current_total_wal_size) {
      std::lock_guard<std::mutex> lock(mu_);
      wal_manager_.SetMaxTotalWALSize(size, current_total_wal_size);
  }

  // Sets the reserved WAL space for a specific column family.
  // Needs the current total size of WAL files to initialize correctly.
  void SetColumnFamilyReservation(uint32_t cf_id, size_t size,
                                  size_t current_total_wal_size) {
       std::lock_guard<std::mutex> lock(mu_);
       wal_manager_.SetColumnFamilyReservation(cf_id, size, current_total_wal_size);
  }

  // === Functions interacting with WALManager ===

  // Checks if a WAL write is permissible based on reservations and limits.
  // Returns OK status if allowed, MemoryLimit status if flush is needed.
  Status CheckCanWriteWAL(uint32_t cf_id, size_t write_size) {
      std::lock_guard<std::mutex> lock(mu_);
      return wal_manager_.CheckCanWriteWAL(cf_id, write_size);
  }

  // Records WAL space usage. Call this after a successful WAL write.
  void AttributeWALSpace(uint32_t cf_id, size_t size) {
       std::lock_guard<std::mutex> lock(mu_);
       wal_manager_.AttributeWALSpace(cf_id, size);
  }

  // Removes WAL space usage attribution. Call this after WAL space associated
  // with a CF is no longer needed (e.g., after memtable flush completes and
  // WAL can potentially be purged).
  void DeattributeWALSpace(uint32_t cf_id, size_t size) {
      std::lock_guard<std::mutex> lock(mu_);
      wal_manager_.DeattributeWALSpace(cf_id, size);
  }

  // Checks if enough global space is available according to reservations.
  // Returns the number of bytes by which the global usage exceeds the allowed
  // limit. Returns 0 if within limits or if the WALManager is disabled.
  // A non-zero return value indicates that flushes of oldest CFs might be needed.
  size_t CheckEnsureReservedSpaceAvailable() {
      std::lock_guard<std::mutex> lock(mu_);
      return wal_manager_.CheckEnsureReservedSpaceAvailable();
  }

  // Logs the current state and usage of the WAL Manager. Returns a string
  // containing the log message.
  std::string LogWALManagerUsage() {
      std::lock_guard<std::mutex> lock(mu_);
      // Caller is responsible for logging the returned string e.g. ROCKS_LOG_INFO(log, "%s", wbm->LogWALManagerUsage().c_str());
      return wal_manager_.LogUsage();
  }

 private:
  std::atomic<size_t> buffer_size_;
  std::atomic<size_t> mutable_limit_;

  // Global Pool
  size_t global_size_;
  std::atomic<size_t> global_used_;
  std::vector<std::atomic<size_t>> per_client_global_used_;

  // Steady Reservation Pool
  size_t steady_res_size_;
  std::vector<bool> per_client_is_steady_;
  std::atomic<size_t> steady_res_used_;
  std::vector<std::atomic<size_t>> per_client_steady_res_used_;

  std::vector<size_t> per_client_buffer_size_;

  std::atomic<size_t> memory_active_;
  std::shared_ptr<CacheReservationManager> cache_res_mgr_;
  std::mutex cache_res_mgr_mu_;

  std::vector<std::list<StallInterface*>> per_client_queue_;
  
  // Protects the per-client queues and stall_active_ flags.
  std::mutex mu_;
  std::atomic<bool> allow_stall_;
  // Value should only be changed by BeginWriteStall() and MaybeEndWriteStall()
  // while holding mu_, but it can be read without a lock.
  std::vector<std::atomic<bool>> per_client_stall_active_;

  std::ofstream mt_log_file_;

  std::vector<std::atomic<size_t>> per_client_stall_count_;
  std::atomic<int> total_stall_count_;

  // Manages WAL space attribution and limits based on reservations.
  WALManager wal_manager_;

  void ReserveMemWithCache(size_t mem);
  void FreeMemWithCache(size_t mem);
};

}  // namespace ROCKSDB_NAMESPACE
