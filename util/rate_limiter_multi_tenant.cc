//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <algorithm>
#include <execinfo.h>
#include <iostream>
#include <limits>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "monitoring/statistics_impl.h"
#include "port/port.h"
#include "rocksdb/system_clock.h"
#include "test_util/sync_point.h"
#include "util/aligned_buffer.h"
#include "util/rate_limiter_multi_tenant_impl.h"
#include "rocksdb/tg_thread_local.h"

namespace ROCKSDB_NAMESPACE {

size_t RateLimiter::RequestToken(size_t bytes, size_t alignment,
                                 Env::IOPriority io_priority, Statistics* stats,
                                 RateLimiter::OpType op_type) {
  if (io_priority < Env::IO_TOTAL && IsRateLimited(op_type)) {
    bytes = std::min(bytes, static_cast<size_t>(GetSingleBurstBytes(op_type)));

    if (alignment > 0) {
      // Here we may actually require more than burst and block
      // as we can not write/read less than one page at a time on direct I/O
      // thus we do not want to be strictly constrained by burst
      bytes = std::max(alignment, TruncateToPageBoundary(alignment, bytes));
    }
    Request(bytes, io_priority, stats, op_type);
  }
  return bytes;
}

// Pending request.
struct MultiTenantRateLimiter::Req {
  explicit Req(int64_t _bytes, port::Mutex* _mu)
      : request_bytes(_bytes), bytes(_bytes), cv(_mu) {}
  int64_t request_bytes;
  int64_t bytes;
  port::CondVar cv;
};

// Key for map tracking pending requests.
struct MultiTenantRateLimiter::ReqKey {
  int client_id;
  Env::IOPriority priority;

  // Constructor
  ReqKey(int c_id, Env::IOPriority pri)
      : client_id(c_id), priority(pri) {}

  bool operator<(const ReqKey& other) const {
    if (client_id != other.client_id) {
      return client_id < other.client_id;
    }
    return priority < other.priority;
  }
};

MultiTenantRateLimiter::MultiTenantRateLimiter(
    int num_clients, std::vector<int64_t> bytes_per_sec,
    std::vector<int64_t> read_bytes_per_sec, int64_t refill_period_us,
    int32_t fairness, RateLimiter::Mode mode,
    const std::shared_ptr<SystemClock>& clock, int64_t single_burst_bytes)
    : RateLimiter(mode),
      refill_period_us_(refill_period_us),
      raw_single_burst_bytes_(single_burst_bytes),
      clock_(clock),
      stop_(false),
      exit_cv_(&request_mutex_),
      requests_to_wait_(0),
      fairness_(fairness > 100 ? 100 : fairness),
      rnd_((uint32_t)time(nullptr)),
      num_clients_(num_clients),
      rate_bytes_per_sec_(num_clients),
      refill_bytes_per_period_(num_clients),
      available_bytes_(num_clients),
      client_mutexes_(num_clients),
      wait_until_refill_pending_(num_clients),
      next_refill_us_(num_clients),
      mode_(mode) {
  // Initialize per-priority counters.
  for (int i = Env::IO_LOW; i < Env::IO_TOTAL; ++i) {
    total_requests_[i] = 0;
    total_bytes_through_[i] = 0;
  }
  // Initialize per-client data structures.
  for (int i = 0; i < num_clients_; ++i) {
    rate_bytes_per_sec_[i].store(bytes_per_sec[i], std::memory_order_relaxed);
    refill_bytes_per_period_[i].store(
        CalculateRefillBytesPerPeriodLocked(bytes_per_sec[i]),
        std::memory_order_relaxed);
    available_bytes_[i].store(refill_bytes_per_period_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    wait_until_refill_pending_[i].store(false, std::memory_order_relaxed);
    next_refill_us_[i] = NowMicrosMonotonicLocked();

    calls_per_client_.emplace_back(0);
    bytes_per_client_.emplace_back(0);
  }
  // Create (empty) queue for each client-priority pair.
  for (int c_id = 0; c_id < num_clients_; ++c_id) {
    for (int pri = Env::IO_LOW; pri < Env::IO_TOTAL; ++pri) {
      ReqKey key(c_id, static_cast<Env::IOPriority>(pri));
      request_queue_map_[key] = std::deque<Req*>();
    }
  }

  // TODO: Create a separate (read) rate limiter if needed
  if (read_bytes_per_sec.size() > 0) {
    read_rate_limiter_ = NewMultiTenantRateLimiter(
        num_clients,
        read_bytes_per_sec,  // <rate_limit> MB/s rate limit
        /* read_rate_limit = */ {},  // Don't create another rate limiter.
        refill_period_us,        // Refill period
        fairness,                // Fairness (default)
        rocksdb::RateLimiter::Mode::kReadsOnly, // Apply only to reads
        single_burst_bytes
    );
  }
}

MultiTenantRateLimiter::~MultiTenantRateLimiter() {
  stop_.store(true, std::memory_order_relaxed);
  // Wake up all waiting threads
  for (int c_id = 0; c_id < num_clients_; ++c_id) {
    MutexLock g_lock(&client_mutexes_[c_id]);
    for (int pri = Env::IO_LOW; pri < Env::IO_TOTAL; ++pri) {
      ReqKey req_key(c_id, static_cast<Env::IOPriority>(pri));
      auto& queue = request_queue_map_[req_key];
      for (auto& req : queue) {
        req->cv.Signal();
      }
    }
  }
  // Wait for all requests to complete
  MutexLock g(&request_mutex_);
  while (requests_to_wait_ > 0) {
    exit_cv_.Wait();
  }
}

// This API allows user to dynamically change rate limiter's bytes per second.
void MultiTenantRateLimiter::SetBytesPerSecond(std::vector<int64_t> bytes_per_second) {
  MutexLock g(&request_mutex_);
  SetBytesPerSecondLocked(bytes_per_second);
}

void MultiTenantRateLimiter::SetBytesPerSecondLocked(std::vector<int64_t> bytes_per_second) {
  for (size_t i = 0; i < bytes_per_second.size(); ++i) {
    rate_bytes_per_sec_[i].store(bytes_per_second[i], std::memory_order_relaxed);
    refill_bytes_per_period_[i].store(
        CalculateRefillBytesPerPeriodLocked(bytes_per_second[i]),
        std::memory_order_relaxed);
  }
}

void MultiTenantRateLimiter::SetBytesPerSecond(int64_t bytes_per_second) {
  (void)bytes_per_second;
  // Deprecated function
}

void MultiTenantRateLimiter::SetBytesPerSecondLocked(int64_t bytes_per_second) {
  (void)bytes_per_second;
  // Deprecated function
}

int64_t MultiTenantRateLimiter::GetSingleBurstBytes() const {
  int client_id = TG_GetThreadMetadata().client_id;
  return GetSingleBurstBytes(client_id);
}

int64_t MultiTenantRateLimiter::GetSingleBurstBytes(OpType op_type) const {
  if (op_type == RateLimiter::OpType::kRead) {
    if (read_rate_limiter_ != nullptr) {
      return read_rate_limiter_->GetSingleBurstBytes();
    } else {
      return 0;
    }
  }
  return GetSingleBurstBytes();
}

int64_t MultiTenantRateLimiter::GetSingleBurstBytes(int client_id) const {
  // return 2 * 1024 * 1024;  // 2 MB
  return refill_bytes_per_period_[client_id].load(std::memory_order_relaxed);
}

Status MultiTenantRateLimiter::SetSingleBurstBytes(int64_t single_burst_bytes) {
  if (single_burst_bytes < 0) {
    return Status::InvalidArgument(
        "`single_burst_bytes` must be greater than or equal to 0");
  }

  MutexLock g(&request_mutex_);
  raw_single_burst_bytes_.store(single_burst_bytes, std::memory_order_relaxed);
  return Status::OK();
}

Status MultiTenantRateLimiter::GetTotalPendingRequests(
    int64_t* total_pending_requests,
    const Env::IOPriority pri) const {
  assert(total_pending_requests != nullptr);
  int64_t total_pending_requests_sum = 0;
  for (const auto& pair : request_queue_map_) {
    if (pri == Env::IO_TOTAL || pair.first.priority == pri) {
      total_pending_requests_sum += static_cast<int64_t>(pair.second.size());
    }
  }
  *total_pending_requests = total_pending_requests_sum;
  return Status::OK();
}

void MultiTenantRateLimiter::TGprintStackTrace() {
  void* array[20];
  size_t size;
  char** strings;
  size_t i;

  size = backtrace(array, 20);
  strings = backtrace_symbols(array, size);

  printf("Obtained %zd stack frames.\n", size);

  for (i = 0; i < size; i++)
    printf("%s\n", strings[i]);

  free(strings);
}

void MultiTenantRateLimiter::Request(int64_t bytes, const Env::IOPriority pri,
                                     Statistics* stats, OpType op_type) {
  if (op_type == RateLimiter::OpType::kRead) {
    if (read_rate_limiter_ != nullptr) {
      read_rate_limiter_->Request(bytes, pri, stats);
    }
    return;
  } else {
    Request(bytes, pri, stats);
  }
}

void MultiTenantRateLimiter::Request(int64_t bytes, const Env::IOPriority pri,
                                     Statistics* stats) {
  // Extract client ID from thread-local metadata.
  int client_id = TG_GetThreadMetadata().client_id;

  if (client_id == -2) {
    compaction_calls_.fetch_add(1, std::memory_order_relaxed);
    compaction_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return;
  }
  if (client_id < 0) {
    unassigned_calls_.fetch_add(1, std::memory_order_relaxed);
    unassigned_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return;
  }

  calls_per_client_[client_id]++;
  bytes_per_client_[client_id] += bytes;

  assert(bytes <= GetSingleBurstBytes(client_id));
  bytes = std::max(static_cast<int64_t>(0), bytes);
  TEST_SYNC_POINT("MultiTenantRateLimiter::Request");
  TEST_SYNC_POINT_CALLBACK("MultiTenantRateLimiter::Request:1",
                           &rate_bytes_per_sec_);

  if (stop_.load(std::memory_order_relaxed)) {
    // Cleanup in progress; exit early.
    return;
  }

  // total_requests_[pri] += 1;
  // int64_t available = available_bytes_[client_id];
  // if (bytes <= available) {
  //   bytes = 0;
  //   available -= bytes;
  // } else {
  //   bytes -= available;
  //   available = 0;
  // }
  // available_bytes_[client_id] = available;

  total_requests_[pri].fetch_add(1, std::memory_order_relaxed);

  int64_t available = available_bytes_[client_id].load(std::memory_order_relaxed);
  if (available > 0) {
    int64_t bytes_to_consume = std::min(available, bytes);
    int64_t expected = available;
    while (!available_bytes_[client_id].compare_exchange_weak(
        expected, expected - bytes_to_consume, std::memory_order_relaxed)) {
      // Retry if the value changed
      bytes_to_consume = std::min(expected, bytes);
    }
    total_bytes_through_[pri].fetch_add(bytes_to_consume, std::memory_order_relaxed);
    bytes -= bytes_to_consume;
  }

  if (bytes == 0) {
    // Request fully satisfied without locking
    return;
  }

  // Acquire the per-client mutex
  MutexLock g_lock(&client_mutexes_[client_id]);

  if (stop_.load(std::memory_order_relaxed)) {
    return;
  }

  // Enqueue the remaining request
  Req req(bytes, &client_mutexes_[client_id]);
  request_queue_map_[ReqKey(client_id, pri)].push_back(&req);
  requests_to_wait_.fetch_add(1, std::memory_order_relaxed);

  // A thread representing a queued request coordinates with other such threads.
  do {
    int64_t time_until_refill_us = next_refill_us_[client_id] - NowMicrosMonotonicLocked();
    if (time_until_refill_us > 0) {
      if (wait_until_refill_pending_[client_id].load(std::memory_order_relaxed)) {
        // Another thread for this client is already waiting; wait for notification
        req.cv.Wait();
      } else {
        // This thread will wait until the next refill time
        wait_until_refill_pending_[client_id].store(true, std::memory_order_relaxed);
        int64_t wait_until = clock_->NowMicros() + time_until_refill_us;
        RecordTick(stats, NUMBER_RATE_LIMITER_DRAINS);
        clock_->TimedWait(&req.cv, std::chrono::microseconds(wait_until));
        wait_until_refill_pending_[client_id].store(false, std::memory_order_relaxed);
      }
    } else {
      // Refill tokens and grant requests for this client
      RefillBytesAndGrantRequestsForClientLocked(client_id);
    }

    if (req.request_bytes == 0) {
      // Request has been granted
      break;
    }
  } while (!stop_.load(std::memory_order_relaxed));

  if (stop_.load(std::memory_order_relaxed)) {
    // Cleanup in progress
    requests_to_wait_.fetch_sub(1, std::memory_order_relaxed);
    exit_cv_.Signal();
  }
}

void MultiTenantRateLimiter::RefillBytesAndGrantRequestsForClientLocked(int client_id) {
  // Update next_refill_us_ for this client
  next_refill_us_[client_id] = NowMicrosMonotonicLocked() + refill_period_us_;

  // Refill tokens for this client
  int64_t refill_bytes = refill_bytes_per_period_[client_id].load(std::memory_order_relaxed);
  int64_t available = available_bytes_[client_id].load(std::memory_order_relaxed);
  int64_t new_available = std::min(available + refill_bytes, GetSingleBurstBytes(client_id));
  available_bytes_[client_id].store(new_available, std::memory_order_relaxed);

  // Process queues for this client
  for (int priority = Env::IO_TOTAL - 1; priority >= Env::IO_LOW; --priority) {
    ReqKey req_key(client_id, static_cast<Env::IOPriority>(priority));
    auto& queue = request_queue_map_[req_key];
    while (!queue.empty()) {
      auto* next_req = queue.front();
      int64_t req_bytes = next_req->request_bytes;
      int64_t client_available = available_bytes_[client_id].load(std::memory_order_relaxed);
      if (client_available < req_bytes) {
        // Not enough tokens to grant this request
        break;
      }
      // Consume available bytes
      available_bytes_[client_id].fetch_sub(req_bytes, std::memory_order_relaxed);
      next_req->request_bytes = 0;
      total_bytes_through_[priority].fetch_add(next_req->bytes, std::memory_order_relaxed);
      queue.pop_front();
      requests_to_wait_.fetch_sub(1, std::memory_order_relaxed);

      // Signal the thread
      next_req->cv.Signal();
    }
  }
}

int64_t MultiTenantRateLimiter::CalculateRefillBytesPerPeriodLocked(
    int64_t rate_bytes_per_sec) {
  if (std::numeric_limits<int64_t>::max() / rate_bytes_per_sec <
      refill_period_us_) {
    // Avoid unexpected result in the overflow case. The result now is still
    // inaccurate but is a number that is large enough.
    return std::numeric_limits<int64_t>::max() / kMicrosecondsPerSecond;
  } else {
    return rate_bytes_per_sec * refill_period_us_ / kMicrosecondsPerSecond;
  }
}

int64_t MultiTenantRateLimiter::GetTotalBytesThrough(
    const Env::IOPriority pri) const {
  if (pri == Env::IO_TOTAL) {
    int64_t total_bytes_through_sum = 0;
    for (int i = Env::IO_LOW; i < Env::IO_TOTAL; ++i) {
      total_bytes_through_sum += total_bytes_through_[i].load(std::memory_order_relaxed);
    }
    return total_bytes_through_sum;
  }
  return total_bytes_through_[pri].load(std::memory_order_relaxed);
}

int64_t MultiTenantRateLimiter::GetTotalRequests(
    const Env::IOPriority pri) const {
  if (pri == Env::IO_TOTAL) {
    int64_t total_requests_sum = 0;
    for (int i = Env::IO_LOW; i < Env::IO_TOTAL; ++i) {
      total_requests_sum += total_requests_[i].load(std::memory_order_relaxed);
    }
    return total_requests_sum;
  }
  return total_requests_[pri].load(std::memory_order_relaxed);
}

int64_t MultiTenantRateLimiter::GetTotalBytesThroughForClient(int client_id) const {
  return bytes_per_client_[client_id];
}

RateLimiter* NewMultiTenantRateLimiter(
    int num_clients /* = 1 */,
    std::vector<int64_t> bytes_per_sec,
    std::vector<int64_t> read_bytes_per_sec,
    int64_t refill_period_us /* = 100 * 1000 */,
    int32_t fairness /* = 10 */,
    RateLimiter::Mode mode /* = RateLimiter::Mode::kWritesOnly */,
    int64_t single_burst_bytes /* = 0 */) {
  for (const int64_t limit : bytes_per_sec) {
    assert(limit > 0);
  }
  assert(refill_period_us > 0);
  assert(fairness > 0);
  std::unique_ptr<RateLimiter> limiter(new MultiTenantRateLimiter(
      num_clients, bytes_per_sec, read_bytes_per_sec, refill_period_us, fairness, mode,
      SystemClock::Default(), single_burst_bytes));
  return limiter.release();
}

}  // namespace ROCKSDB_NAMESPACE
