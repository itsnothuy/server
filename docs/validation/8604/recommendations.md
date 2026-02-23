# Recommendations

> **Factified 2026-02-23** — All recommendations updated with precise code
> citations and corrected understanding of the readiness architecture.

## Final Recommendation

**Create a new PR superseding PR #431 in python_backend + Open a separate PR in triton core for server-level readiness.**

This is a two-track approach with corrections based on factified validation findings.

---

## Track 1: Python Backend — New PR Superseding PR #431

### Why supersede rather than improve?

PR #431 has 6 critical implementation issues (concurrency, string matching, request retry safety, no rate limiting, non-functional tests, no readiness fix). The issues are fundamental enough that patching PR #431 would require rewriting most of its code. A clean PR that acknowledges PR #431's contribution is more appropriate.

### Required Implementation

#### 1. Add concurrency protection (CRITICAL)

```cpp
// In python_be.h, add to ModelInstanceState private:
std::mutex restart_mutex_;
std::atomic<bool> restart_in_progress_{false};
```

The restart path in `TRITONBACKEND_ModelInstanceExecute` must be guarded:
```cpp
if (error != nullptr && !instance_state->IsStubProcessAlive()) {
  std::unique_lock<std::mutex> lock(instance_state->restart_mutex_, std::try_to_lock);
  if (lock.owns_lock() && !instance_state->restart_in_progress_.exchange(true)) {
    TRITONSERVER_ErrorDelete(error);
    error = instance_state->RestartStubProcess();
    instance_state->restart_in_progress_ = false;
    // Do NOT retry current requests — they already failed
    // The stub is now ready for the NEXT request
  }
}
```

#### 2. Replace string matching with proper detection (CRITICAL)

Instead of matching error text, check stub health directly:
```cpp
if (error != nullptr && !instance_state->IsStubProcessAlive()) {
  // Stub is confirmed dead — attempt restart
}
```

This catches ALL failure modes:
- `"Stub process is not healthy."` from `SendMessageToStub()`
- `"Failed to obtain the health mutex."` from `SendMessageToStub()`
- Any error from `ReceiveMessageFromStub()` when stub is dead

**Do NOT retry the current request.** The first `ProcessRequests()` call may have:
- Created response factories via `TRITONBACKEND_ResponseFactoryNew()`
- Written to shared memory regions now owned by the old stub
- Partially populated `infer_requests` with pointers to invalid memory

Retrying creates double-free / use-after-free risks. Instead, fail the current batch and have the stub ready for subsequent requests.

#### 3. Fix RestartStubProcess() for crash context (CRITICAL)

```cpp
TRITONSERVER_Error* ModelInstanceState::RestartStubProcess() {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
    (std::string("Restarting unhealthy stub process for instance '") +
     Name() + "' (restart #" + std::to_string(restart_count_ + 1) + ")").c_str());
  try {
    if (Stub()) {
      // Do NOT call UpdateHealth/IsHealthy — the stub is dead or unresponsive.
      // Do NOT wait on thread_pool — tasks may be stuck on dead IPC.
      // Kill the stub process first (handles zombies via SIGKILL).
      Stub()->TerminateStub();
      TerminateMonitor();
      Stub()->ClearQueues();
      Stub().reset();
    }
    RETURN_IF_ERROR(LaunchStubProcess());
    restart_count_++;
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
      (std::string("Successfully restarted stub process for instance '") +
       Name() + "'").c_str());
    return nullptr;
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
      (std::string("Failed to restart stub process for instance '") +
       Name() + "': " + ex.what()).c_str());
  }
}
```

Key differences from PR #431:
1. Skips `UpdateHealth()`/`IsHealthy()` — unsafe when stub crashed holding mutex
2. Skips `thread_pool_->wait()` — tasks may be stuck on dead IPC channels
3. Calls `TerminateStub()` before `TerminateMonitor()` — matches destructor intent
4. Tracks restart count for rate limiting

#### 4. Add restart rate limiting (MEDIUM)

```cpp
// In python_be.h:
std::atomic<int> restart_count_{0};
std::chrono::steady_clock::time_point restart_window_start_{
    std::chrono::steady_clock::now()};
static constexpr int kMaxRestartsPerWindow = 5;
static constexpr int kRestartWindowSeconds = 60;
```

Before restarting, check:
```cpp
bool ModelInstanceState::CanRestart() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      now - restart_window_start_).count();
  if (elapsed > kRestartWindowSeconds) {
    // Reset window
    restart_count_ = 0;
    restart_window_start_ = now;
  }
  return restart_count_ < kMaxRestartsPerWindow;
}
```

If limit exceeded, log and leave instance permanently unhealthy. This prevents restart storms when the model code itself is fundamentally broken.

#### 5. Add real integration tests (CRITICAL)

Replace the grep-based tests with actual integration tests following the pattern in `qa/L0_backend_python/model_readiness/test.sh`:

```bash
# test_stub_restart.sh
# 1. Start server with kill_stub model
# 2. Verify model initially ready: is_model_ready() == True
# 3. Send inference that kills stub (os._exit(0))
# 4. Wait for restart: poll server logs for "Successfully restarted stub process"
# 5. Send another inference — verify it succeeds
# 6. Verify is_model_ready() == True after restart
```

The Python test class should use `tritonclient.http` and `tritonclient.grpc` to verify behavior end-to-end.

#### 6. Handle decoupled models (LOW — separate follow-up PR)

For decoupled models, before restart:
- Signal the monitor thread to stop processing response-send messages
- Close all outstanding response factories with error responses
- Wait for `thread_pool_` tasks to drain (with timeout, e.g., 5 seconds)

This is complex and can be a separate PR. In the initial PR, decoupled models should still get restart (the core mechanism is the same), but with a log warning that in-flight responses may be lost.

---

## Track 2: Server Core — Fix `/v2/health/ready` Readiness

### The Problem (precisely stated)

`InferenceServer::IsReady()` (core `src/server.cc` L417-457) with `strict_readiness_=true`:
- Calls `model_repository_manager_->ModelStates()` — returns `ModelReadyState` lifecycle enum
- Checks if each model version is `ModelReadyState::READY` (or unloaded)
- Does **NOT** call `model->IsReady()` — never invokes backend instance readiness checks

Meanwhile, `InferenceServer::ModelIsReady()` (core `src/server.cc` L459-487):
- First checks lifecycle state
- **Then** calls `model->IsReady()` → `TritonModel::IsReady()` → iterates instances → `TritonModelInstance::IsReady()` → `TRITONBACKEND_ModelInstanceReady`

The gap: `IsReady()` is lifecycle-only; `ModelIsReady()` is lifecycle + runtime. The fix is to make `IsReady()` also do runtime checks.

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
        // Models in READY lifecycle state: also check runtime readiness
        if (vs.second.first == ModelReadyState::READY) {
          bool model_ready = false;
          Status status = ModelIsReady(
              mv.first.name_, vs.first /* version */, &model_ready);
          if (!status.IsOk() || !model_ready) {
            *ready = false;
            goto strict_done;
          }
        }
      }
    }
  strict_done:;
  }
  return Status::Success;
}
```

### Is this a bugfix or a behavior change?

**This is a bugfix.** The [Triton documentation](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_health.md) states:
> "The ready endpoint indicates whether the server is able to respond to inference requests."

With `--strict-readiness=true`, the server should be ready only when all models can actually serve inference. A model with a dead Python stub cannot serve inference, so returning "ready" is incorrect.

The behavior change is minimal: models that were in `READY` lifecycle state but had unhealthy backend instances were falsely reported as ready. After this fix, they are correctly reported as not ready.

### Considerations

1. **Performance**: Calling `ModelIsReady()` for every model on every `/v2/health/ready` request adds overhead. The `TRITONBACKEND_ModelInstanceReady` call in the Python backend does `waitpid(WNOHANG)` (non-blocking, microsecond-level). The `IsStubProcessAlive()` health mutex check with 1-second timeout is NOT called here — only `StubActive()` is. So overhead is minimal.

2. **Scope**: This change goes in `triton-inference-server/core`, not `triton-inference-server/server`. The server repo has HTTP/gRPC handlers that delegate to the core library.

3. **No flag needed**: Since this is a bugfix (aligning behavior with docs), and since `StubActive()` is a lightweight `waitpid(WNOHANG)` call, no feature flag is needed. The fix is unconditional when `strict_readiness_=true`.

---

## Risk Assessment

| Track | Risk | Probability | Impact | Mitigation |
|-------|------|-------------|--------|------------|
| Track 1: Concurrent restart without mutex | Race condition, crash | High | Critical | Add `restart_mutex_` |
| Track 1: String matching missing failures | Stub stays dead silently | Medium | High | Use `IsStubProcessAlive()` |
| Track 1: Request retry after restart | Double-free, UB | High | Critical | Don't retry; fail current batch |
| Track 2: Runtime readiness overhead | Slow health checks | Low | Low | `waitpid(WNOHANG)` is fast |
| Track 2: False negatives during restart | Transient non-ready | Low | Low | Restart completes in seconds |

---

## Priority Order

1. **P0 (immediate)**: Track 1 — Production-safe stub restart (concurrency, detection, rate limiting, tests)
2. **P1 (soon)**: Track 2 — Server core readiness fix
3. **P2 (follow-up)**: Decoupled model restart handling
4. **P3 (future)**: Restart metrics and observability
