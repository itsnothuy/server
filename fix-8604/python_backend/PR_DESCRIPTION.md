# PR Description: Fix #8604 — Production-Safe Python Backend Stub Restart

## Target Repository
`triton-inference-server/python_backend` → `main`

## Summary

Re-implements the Python backend stub restart mechanism that was removed by PR #360 during the decoupled/non-decoupled pipeline unification. When a Python backend stub process dies (e.g., due to `os._exit()`, SIGSEGV, SIGKILL), the backend now automatically detects the death and restarts the stub, restoring model functionality without manual intervention.

## Problem

**Issue:** [triton-inference-server/server#8604](https://github.com/triton-inference-server/server/issues/8604)
**Related:** [triton-inference-server/server#7230](https://github.com/triton-inference-server/server/issues/7230)

When a Python backend stub process becomes unhealthy or dies:
1. Inference requests fail with "Stub process is not healthy"
2. The stub is never restarted — the TODO comment at `TRITONBACKEND_ModelInstanceExecute` line ~2335 was the only remnant of the old restart code removed by PR #360
3. `/v2/models/{model}/ready` correctly reports unhealthy (via `TRITONBACKEND_ModelInstanceReady` → `StubActive()` → `waitpid(WNOHANG)`)
4. But the model stays permanently broken until the entire server is restarted

## Solution

### Detection
- After `ProcessRequests()` returns an error, check `IsStubProcessAlive()` directly instead of brittle string matching on error messages
- This catches ALL failure modes: "Stub process is not healthy", "Failed to obtain the health mutex", and any `ReceiveMessageFromStub` failure when the stub is dead

### Restart
- `RestartStubProcess()` method: kills the dead stub via `TerminateStub()` (SIGKILL), stops the monitor thread, clears IPC queues, destroys the old `StubLauncher`, then calls `LaunchStubProcess()` which creates a completely fresh stub process
- Does NOT call `UpdateHealth()`/`IsHealthy()` — unsafe when stub crashed holding the health mutex
- Does NOT wait on `thread_pool_` — pending tasks may be stuck on dead IPC channels

### Safety
- **Concurrency protection:** `std::mutex restart_mutex_` with `try_to_lock` + `std::atomic<bool> restart_in_progress_` prevents multiple threads from attempting restart simultaneously
- **Rate limiting:** Maximum 5 restarts per 60-second window (`kMaxRestartsPerWindow`, `kRestartWindowSeconds`). After exceeding the limit, the instance is marked permanently unhealthy with a clear log message
- **No request retry:** The triggering request batch is failed with the original error. The restart ensures *subsequent* requests succeed. This avoids double-free/use-after-free risks from reprocessing partially-consumed request objects

### Comparison with PR #431

This PR supersedes [PR #431](https://github.com/triton-inference-server/python_backend/pull/431) by @paipeline, which has the correct intent but has critical implementation issues:

| Issue | PR #431 | This PR |
|-------|---------|---------|
| Concurrency | No locking | `restart_mutex_` + `restart_in_progress_` atomic |
| Detection | `string::find("Stub process is not healthy")` — misses "Failed to obtain the health mutex" | `IsStubProcessAlive()` — catches all failure modes |
| Request retry | Retries same request batch → double-free risk | Fails current batch, restarts for next |
| Rate limiting | None → infinite restart storms | 5 per 60s window |
| Tests | grep-based static checks on source code | Real integration tests |
| `UpdateHealth()` on dead stub | Called → blocks on dead mutex | Skipped |

## Tests

Integration test (`tests/stub_restart/`) follows the pattern from `qa/L0_backend_python/model_readiness/`:

1. **test_normal_inference** — Baseline: inference works
2. **test_restart_after_kill** — Kill stub via `os._exit(0)` → wait for restart (polling, no sleep) → verify next inference succeeds
3. **test_readiness_after_restart** — Verify `is_model_ready()` returns True after restart
4. **test_restart_rate_limit** — Kill stub >5 times → verify model stays permanently unhealthy

Server log is verified for restart markers:
- `"Detected dead stub process for instance '...'"` (WARN)
- `"Successfully restarted stub process for instance '...'"` (INFO)
- `"Stub restart rate limit exceeded"` (ERROR, when applicable)

## Remaining Gap: `/v2/health/ready`

This PR fixes stub restart in the python_backend. However, `/v2/health/ready` with `--strict-readiness=true` still returns 200 even when the stub is dead, because `InferenceServer::IsReady()` in triton core only checks `ModelReadyState` lifecycle state (not runtime backend readiness). A separate PR to `triton-inference-server/core` is needed to fix this — see linked issue comments.

## Files Changed

| File | Changes |
|------|---------|
| `src/python_be.h` | Add `RestartStubProcess()`, `CanRestart()`, `restart_mutex_`, `restart_in_progress_`, restart rate limiting members |
| `src/python_be.cc` | Add `RestartStubProcess()`, `CanRestart()`. Modify `TRITONBACKEND_ModelInstanceExecute` with restart detection |
| `tests/stub_restart/test.sh` | Integration test shell script |
| `tests/stub_restart/test_stub_restart.py` | Integration test Python client |

## How to Test Locally

```bash
# Build python_backend with the fix
cd python_backend && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$(pwd)/install \
      -DTRITON_ENABLE_GPU=ON \
      -DTRITON_BACKEND_REPO_TAG=main \
      -DTRITON_CORE_REPO_TAG=main \
      -DTRITON_COMMON_REPO_TAG=main ..
make -j$(nproc) install

# Run integration test
cd ../tests/stub_restart
export BACKEND_DIR=/path/to/backends
export NVIDIA_TRITON_SERVER_VERSION=2.67.0
bash test.sh $NVIDIA_TRITON_SERVER_VERSION
```
