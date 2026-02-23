# Recommendations

## Final Recommendation

**Contribute improvements to PR #431 in python_backend + Open a separate PR in triton core for server-level readiness.**

This is a two-track approach matching the structure proposed in our fix files, with corrections based on validation findings.

---

## Track 1: Python Backend — Improve PR #431

### Why improve rather than replace?

PR #431 has the right location and intent but has critical implementation flaws. Contributing improvements is more collaborative and likely to be accepted than a competing PR. If `@paipeline` is unresponsive, fork and submit a refined version.

### Required Improvements

#### 1. Add concurrency protection (CRITICAL)

```cpp
// In python_be.h, add to ModelInstanceState:
std::mutex restart_mutex_;
std::atomic<bool> restart_in_progress_{false};
```

The restart path in `TRITONBACKEND_ModelInstanceExecute` must be guarded:
```cpp
{
  std::lock_guard<std::mutex> lock(instance_state->restart_mutex_);
  if (!instance_state->restart_in_progress_.exchange(true)) {
    error = instance_state->RestartStubProcess();
    instance_state->restart_in_progress_ = false;
  }
}
```

#### 2. Replace string matching with proper detection (CRITICAL)

Instead of matching error text, use a flag or check stub health directly:
```cpp
if (error != nullptr && !instance_state->IsStubProcessAlive()) {
  // Stub is confirmed dead — attempt restart
  TRITONSERVER_ErrorDelete(error);
  error = instance_state->RestartStubProcess();
  // Don't retry the current request — it already failed
  // The stub is now ready for the NEXT request
}
```

**Do NOT retry the current request.** The first `ProcessRequests()` call may have partially consumed request resources (response factories, shared memory allocations). Retrying creates double-free risks. Instead, let the current request fail and have the stub ready for subsequent requests.

#### 3. Add restart rate limiting (MEDIUM)

```cpp
// In python_be.h:
std::atomic<int> restart_count_{0};
std::chrono::steady_clock::time_point restart_window_start_;
static constexpr int kMaxRestartsPerWindow = 5;
static constexpr int kRestartWindowSeconds = 60;
```

If `restart_count_` exceeds `kMaxRestartsPerWindow` within `kRestartWindowSeconds`, stop restarting and log a fatal error. The model instance remains permanently unhealthy.

#### 4. Fix RestartStubProcess() cleanup order (MEDIUM)

Match the destructor pattern but handle the dead-stub case:
```cpp
TRITONSERVER_Error* ModelInstanceState::RestartStubProcess() {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
    (std::string("Restarting unhealthy stub process for instance ") + Name()).c_str());
  try {
    if (Stub()) {
      // Don't call UpdateHealth/IsHealthy — the stub is dead
      // Don't wait on thread_pool — tasks may be stuck on dead IPC
      Stub()->TerminateStub();   // Sends SIGKILL if needed
      TerminateMonitor();        // Stop monitor thread
      Stub()->ClearQueues();     // Clean up shared memory queues
      Stub().reset();            // Destroy StubLauncher
    }
    // LaunchStubProcess creates new Stub, monitor, thread_pool, request_executor
    RETURN_IF_ERROR(LaunchStubProcess());
    return nullptr;
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
      (std::string("Failed to restart stub process: ") + ex.what()).c_str());
  }
}
```

#### 5. Add real integration tests (CRITICAL)

Replace the grep-based tests with actual integration tests. Use the existing pattern from `qa/L0_backend_python/model_readiness/test.sh`:

```bash
# test_stub_restart.sh
# 1. Start server with kill_stub model
# 2. Verify model initially ready
# 3. Send inference that kills stub (os._exit(0))
# 4. Wait for restart (poll logs for "Restarting unhealthy stub")
# 5. Send another inference — verify it succeeds
# 6. Verify /v2/models/kill_stub/ready returns 200
```

#### 6. Handle decoupled models (LOW — can be follow-up)

For decoupled models, before restart:
- Signal the monitor thread to stop processing response-send messages
- Close all outstanding response factories with error responses
- Wait for `thread_pool_` tasks to drain (with timeout)

This is complex and could be a separate PR.

---

## Track 2: Server Core — Fix `/v2/health/ready` Readiness

### The Problem

`InferenceServer::IsReady()` with `strict_readiness_=true` checks `ModelStates()` which only returns lifecycle state (`ModelReadyState::READY`). It does NOT call `model->IsReady()` which would invoke `TRITONBACKEND_ModelInstanceReady`.

### The Fix

Modify `InferenceServer::IsReady()` in `core/src/server.cc`:

```cpp
Status InferenceServer::IsReady(bool* ready) {
  *ready = false;
  if (ready_state_ == ServerReadyState::SERVER_EXITING)
    return Status(Status::Code::UNAVAILABLE, "Server exiting");

  ScopedAtomicIncrement inflight(inflight_request_counter_);
  *ready = (ready_state_ == ServerReadyState::SERVER_READY);

  if (*ready && strict_readiness_) {
    const auto model_versions = model_repository_manager_->ModelStates();
    for (const auto& mv : model_versions) {
      if (mv.second.size() == 0) { *ready = false; goto strict_done; }
      for (const auto& vs : mv.second) {
        if ((vs.second.first != ModelReadyState::READY) &&
            (vs.second.second != "unloaded")) {
          *ready = false; goto strict_done;
        }
      }
      // NEW: Also check runtime instance readiness for READY models
      // This catches cases where the model lifecycle says READY but
      // the backend reports unhealthy (e.g., dead Python stub)
      if (runtime_readiness_check_enabled_) {
        bool model_ready = false;
        auto status = ModelIsReady(mv.first.name_, -1 /* latest */, &model_ready);
        if (!model_ready) { *ready = false; goto strict_done; }
      }
    }
  strict_done:;
  }
  return Status::Success;
}
```

### Considerations

1. **Performance**: Calling `ModelIsReady()` for every model on every `/v2/health/ready` request adds overhead. The `TRITONBACKEND_ModelInstanceReady` call in the Python backend acquires a mutex with 1-second timeout, so this could add latency. Gate behind a flag initially.

2. **Flag**: Add `TRITON_ENABLE_RUNTIME_READINESS_CHECK` environment variable (default: off for backward compatibility, then eventually default on).

3. **Target repo**: This change goes in `triton-inference-server/core`, not `triton-inference-server/server`. The server repo only has HTTP/gRPC handlers that delegate to the core library.

---

## Risk Assessment

| Track | Risk | Probability | Impact | Mitigation |
|-------|------|-------------|--------|------------|
| Track 1: Restart without mutex | Race condition on concurrent requests | High (multi-threaded server) | Crash/UB | Add restart_mutex_ |
| Track 1: String matching | Missed failure modes | Medium | Stub stays dead | Use IsStubProcessAlive() |
| Track 1: Request retry | Double-free | High | Crash | Don't retry; fail current request |
| Track 2: Performance | Slow health checks | Low (sub-second) | Latency | Gate behind flag |
| Track 2: False negatives | Model transiently unhealthy | Low | Flapping readiness | Add hysteresis/grace period |

---

## Priority Order

1. **P0 (immediate)**: Fix PR #431's concurrency and request-retry issues
2. **P0 (immediate)**: Replace PR #431's non-functional tests with real integration tests
3. **P1 (soon)**: Open server core PR for runtime readiness check
4. **P2 (follow-up)**: Add restart rate limiting and metrics
5. **P3 (future)**: Full decoupled model restart handling
