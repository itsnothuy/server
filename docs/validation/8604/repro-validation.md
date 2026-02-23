# Reproduction Validation

> **Factified 2026-02-23** — Code-level validation updated with exact line numbers
> and call chains. Upstream test evidence documented precisely.

## Runtime Execution Constraints

**This validation was performed via code inspection only. No runtime reproduction was executed.**

### Why

- No Docker environment available on the validation host (macOS development machine)
- No NVIDIA GPU available for GPU-accelerated Triton containers
- The `nvcr.io/nvidia/tritonserver` container requires Docker + NVIDIA Container Toolkit
- Building the python_backend from source requires CUDA toolkit, cmake, and Triton core libraries

### What Would Be Needed

To fully reproduce issue #8604:

```bash
# 1. Start Triton with kill_stub model
docker run --gpus=all -it --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $PWD/model_repo:/models \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models --strict-readiness=true --log-verbose=1

# 2. Normal inference (should succeed)
python3 -c "
import tritonclient.http as httpclient
import numpy as np
client = httpclient.InferenceServerClient('localhost:8000')
inp = httpclient.InferInput('INPUT', [1], 'INT32')
inp.set_data_from_numpy(np.array([1], dtype=np.int32))
result = client.infer('kill_stub', [inp])
print('Output:', result.as_numpy('OUTPUT'))
"

# 3. Kill-stub inference (triggers os._exit(0))
python3 -c "
import tritonclient.http as httpclient
import numpy as np
client = httpclient.InferenceServerClient('localhost:8000')
inp = httpclient.InferInput('INPUT', [1], 'INT32')
inp.set_data_from_numpy(np.array([0], dtype=np.int32))
try:
    result = client.infer('kill_stub', [inp])
except Exception as e:
    print('Expected error:', e)
"

# 4. Check endpoints — THE KEY BEHAVIOR DIFFERENCE
curl -v localhost:8000/v2/health/live              # Expected: 200 (correct)
curl -v localhost:8000/v2/health/ready             # Expected: 200 (BUG — should be non-200 with strict-readiness)
curl -v localhost:8000/v2/models/kill_stub/ready   # Expected: non-200 (ALREADY WORKS)

# 5. Check zombie process
docker exec <container> ps aux | grep triton_python_backend_stub
# Expected: <defunct> or Z state (until waitpid reaps it)
```

## Code-Level Validation (Performed)

### Validated via upstream code inspection:

#### 1. `TRITONBACKEND_ModelInstanceReady` correctly detects dead stubs

- **File:** `python_backend/src/python_be.cc` lines 2418-2437
- Calls `Stub()->StubActive()` which uses `waitpid(stub_pid_, &status, WNOHANG)` (stub_launcher.cc L740-769)
- Returns error `"Stub process 'X' is not healthy."` when stub is dead
- **Confirmed by existing upstream test** at `qa/L0_backend_python/model_readiness/test.sh`

#### 2. `/v2/models/{model}/ready` correctly reports unhealthy after stub death

Full call chain (each step verified in source):
```
HandleModelReady (server/src/http_server.cc)
  → TRITONSERVER_ServerModelIsReady (core/src/tritonserver.cc)
    → InferenceServer::ModelIsReady (core/src/server.cc L459-487)
      → checks ModelReadyState (lifecycle) — must be READY
      → model->IsReady() (core/src/backend_model.cc L295-306)
        → instance->IsReady() (for each instance)
          → TritonModelInstance::IsReady (core/src/backend_model_instance.cc L594-617)
            → TRITONBACKEND_ModelInstanceReady (python_backend fn ptr)
              → Stub()->StubActive() → waitpid(WNOHANG)
```

Each step is **non-cached**, evaluated fresh per call. No caching at any layer.

#### 3. `/v2/health/ready` does NOT detect dead stubs (THE BUG)

Full call chain:
```
HandleServerHealth (server/src/http_server.cc)
  → TRITONSERVER_ServerIsReady (core/src/tritonserver.cc)
    → InferenceServer::IsReady (core/src/server.cc L417-457)
      → checks ready_state_ == SERVER_READY
      → if strict_readiness_: model_repository_manager_->ModelStates()
        → only checks ModelReadyState lifecycle enum
        → DOES NOT call model->IsReady()
        → DOES NOT invoke any backend checks
```

A dead stub does NOT change `ModelReadyState` — the model was loaded successfully and remains in `READY` lifecycle state. `IsReady()` sees `READY` and reports the server as ready.

#### 4. `SendMessageToStub()` error paths

- **File:** `python_backend/src/python_be.cc` lines 1072-1104
- **Error 1:** `"Failed to obtain the health mutex."` — when `bi::scoped_lock` times out (1000ms)
- **Error 2:** `"Stub process is not healthy."` — when `StubMessageQueue()->Push()` fails AND `IsStubProcessAlive()` returns false
- Both errors propagate as `TRITONSERVER_ERROR_INTERNAL` to `ProcessRequests()` caller

#### 5. No restart logic exists on `main` branch

- Only orphaned TODO comment at lines 2335-2337
- No `restart` variable declared
- No `RestartStub` or `RestartStubProcess` method exists
- No restart-related members in `ModelInstanceState` class

### Existing upstream test evidence

The test at `qa/L0_backend_python/model_readiness/test.sh` proves:
- After killing the stub with SIGSEGV (11) or SIGKILL (9)
- `is_model_ready()` returns False on both HTTP and gRPC clients
- Server logs contain `"Model '${MODEL_NAME}' version 1 is not ready: Stub process '${MODEL_NAME}_0_0' is not healthy."`
- Test expects exactly 2 error occurrences (HTTP check + gRPC check)

**Critical gap in upstream test:** The test does NOT check `is_server_ready()` or `/v2/health/ready`. It only validates per-model readiness, not server-level readiness.

## Conclusion

The reproduction described in the fix files is **consistent with code analysis** but has an important nuance: `/v2/models/{model}/ready` already works correctly (contrary to what the fix files imply). The actual bug is specifically in `/v2/health/ready` with `--strict-readiness=true`, which uses lifecycle-only checks and never invokes backend readiness.
