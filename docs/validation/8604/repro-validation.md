# Reproduction Validation

## Runtime Execution Constraints

**This validation was performed via code inspection only. No runtime reproduction was executed.**

### Why

- No Docker environment available on the validation host (macOS development machine)
- No NVIDIA GPU available for GPU-accelerated Triton containers
- The `nvcr.io/nvidia/tritonserver:24.05-py3` container requires Docker + NVIDIA Container Toolkit
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

# 4. Check endpoints
curl -v localhost:8000/v2/health/live       # Expected: 200 (correct - server is alive)
curl -v localhost:8000/v2/health/ready      # Expected: 200 (BUG - should be non-200)
curl -v localhost:8000/v2/models/kill_stub/ready  # Expected: non-200 (already works)

# 5. Check zombie process
docker exec <container> ps aux | grep triton_python_backend_stub
# Expected: <defunct> or Z state
```

## Code-Level Validation (Performed)

### Validated via upstream code inspection:

1. **`TRITONBACKEND_ModelInstanceReady`** correctly detects dead stubs:
   - Calls `Stub()->StubActive()` which uses `waitpid(WNOHANG)`
   - Returns error `"Stub process 'X' is not healthy."` when stub is dead
   - **Confirmed by existing upstream test** at `qa/L0_backend_python/model_readiness/test.sh`

2. **`/v2/models/{model}/ready`** correctly reports unhealthy after stub death:
   - Call chain: `HandleModelReady` → `TRITONSERVER_ServerModelIsReady` → `ModelIsReady` → `model->IsReady()` → `instance->IsReady()` → `TRITONBACKEND_ModelInstanceReady`
   - Each step is non-cached, evaluated fresh per call

3. **`/v2/health/ready`** does NOT detect dead stubs (the bug):
   - Call chain: `HandleServerHealth` → `TRITONSERVER_ServerIsReady` → `IsReady()`
   - `IsReady()` with `strict_readiness_=true` calls `ModelStates()` which only checks `ModelReadyState` enum (lifecycle state: READY/LOADING/etc.)
   - A dead stub does NOT change the lifecycle state — the model was loaded successfully and remains in `READY` state
   - `IsReady()` does NOT call `model->IsReady()` per-instance

4. **`SendMessageToStub()`** produces "Stub process is not healthy." error when stub is dead:
   - After message push timeout, checks `IsStubProcessAlive()` which tries health mutex
   - If mutex cannot be acquired or `stub_health` is false, returns error

5. **No restart logic exists on `main` branch**:
   - Only a vestigial TODO comment remains
   - No `restart` variable is declared
   - No `RestartStub` method exists

### Existing upstream test evidence:

The test at `qa/L0_backend_python/model_readiness/test.sh` proves that:
- After killing the stub with SIGSEGV (11) or SIGKILL (9)
- `/v2/models/{model}/ready` reports NOT ready
- Server logs contain `"Model '...' version 1 is not ready: Stub process '..._0_0' is not healthy."`
- The test expects exactly 2 error occurrences (HTTP + gRPC checks)

This test validates per-model readiness but does NOT test server-level readiness (`/v2/health/ready`).

## Conclusion

The reproduction described in the fix files is consistent with the code analysis. The specific gap — `/v2/health/ready` remaining 200 while `/v2/models/{model}/ready` correctly reports unhealthy — is architecturally confirmed through code inspection of the two different readiness call chains in triton core.
