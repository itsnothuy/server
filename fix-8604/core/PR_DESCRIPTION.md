# Track B: Fix /v2/health/ready to Reflect Python Backend Health

## PR Title
**fix: server readiness endpoint detects dead Python backend stubs (#8604)**

## Summary

Fixes the server-level readiness endpoint (`/v2/health/ready`) to reflect
backend runtime health when `--strict-readiness=true` (the default).

### The Bug

When a Python backend stub process dies (crash, OOM kill, segfault):

| Endpoint | Before Fix | After Fix |
|----------|-----------|-----------|
| `/v2/models/{model}/ready` | ✅ Returns 503 (correct) | ✅ Returns 503 (correct) |
| `/v2/health/ready` | ❌ Returns 200 (WRONG) | ✅ Returns 503 (correct) |

The per-model endpoint already works because it calls `ModelIsReady()` →
`model->IsReady()` → `TRITONBACKEND_ModelInstanceReady` → `IsStubProcessAlive()`.

The server-level endpoint was broken because `InferenceServer::IsReady()`
only checked lifecycle state via `ModelStates()`, which remains `READY`
even when the underlying stub process is dead.

### The Fix

In `InferenceServer::IsReady()` (src/server.cc), for each model in `READY`
lifecycle state, additionally call `ModelIsReady()` to verify the backend
reports the model is actually operational:

```cpp
// EXISTING: lifecycle check
if (ms.second.first != ModelReadyState::READY) {
    // not ready...
}
// NEW: runtime health check for models in READY lifecycle state
else {
    bool model_ready = false;
    Status s = ModelIsReady(ms.first, ms.second.second, &model_ready);
    if (!s.IsOk() || !model_ready) {
        // Backend reports model is not actually ready (e.g., dead stub)
        ready_state = false;
        goto done;
    }
}
```

### Why This Is Safe

1. **Performance**: `ModelIsReady()` → `TRITONBACKEND_ModelInstanceReady` →
   `IsStubProcessAlive()` → `waitpid(WNOHANG)` is a non-blocking syscall
   (microseconds). Health checks are infrequent (typically every 5-30s).

2. **Backward compatibility**: Only affects `--strict-readiness=true` path.
   When `strict_readiness_=false`, the new code is never reached.

3. **Non-Python backends**: Backends that don't implement
   `TRITONBACKEND_ModelInstanceReady` return `true` by default, so they are
   unaffected.

## Related Issues

- Fixes #8604 (Python backend: dead stub invisible to /v2/health/ready)
- Related: #7230 (model readiness improvements)
- Related: #7588 (instance death handling)

## Type of Change

- [x] Bug fix (non-breaking change that fixes incorrect behavior)
- [ ] New feature
- [ ] Breaking change

## Test Plan

### New Tests
- `fix-8604/core/tests/server_readiness/test.sh` — Integration test that:
  1. Starts Triton with a Python model + `--strict-readiness=true`
  2. Verifies `/v2/health/ready` returns 200 when healthy
  3. Kills the stub process
  4. Verifies `/v2/health/ready` returns non-200 (the #8604 fix)
  5. Verifies `/v2/models/.../ready` also returns non-200 (baseline)
  6. Restarts with `--strict-readiness=false`, kills stub, verifies 200

### Existing Tests
- `qa/L0_backend_python/model_readiness/test.sh` — Tests per-model readiness
  (already passes; our change doesn't affect this path)

## Files Changed

| File | Description |
|------|-------------|
| `src/server.cc` | Add `ModelIsReady()` call in `InferenceServer::IsReady()` strict path |
| `tests/server_readiness/test.sh` | Integration test (shell) |
| `tests/server_readiness/test_server_readiness.py` | Integration test (Python) |
