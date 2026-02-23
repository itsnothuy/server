# PR Plan

## Overview

Two PRs are needed to fully resolve issue #8604:

| PR | Repository | Scope | Priority |
|----|-----------|-------|----------|
| PR-A | `triton-inference-server/python_backend` | Stub restart (improve PR #431) | P0 |
| PR-B | `triton-inference-server/core` | Server-level readiness fix | P1 |

---

## PR-A: Python Backend Stub Restart

### Target

Contribute improvements to existing [PR #431](https://github.com/triton-inference-server/python_backend/pull/431) or open a refined replacement.

### Changed Files

| File | Changes |
|------|---------|
| `src/python_be.h` | Add `RestartStubProcess()`, `restart_mutex_`, `restart_count_`, `restart_in_progress_` |
| `src/python_be.cc` | Implement restart logic in `TRITONBACKEND_ModelInstanceExecute` and `RestartStubProcess()` |
| `tests/stub_restart/` | New integration test directory |

### Implementation Details

#### `python_be.h` additions

```cpp
class ModelInstanceState : public BackendModelInstance {
  // ... existing members ...

  // Restart unhealthy stub process with rate limiting
  TRITONSERVER_Error* RestartStubProcess();

 private:
  // Concurrency protection for restart
  std::mutex restart_mutex_;
  std::atomic<bool> restart_in_progress_{false};

  // Restart rate limiting
  std::atomic<int> restart_count_{0};
  std::chrono::steady_clock::time_point restart_window_start_{
      std::chrono::steady_clock::now()};
  static constexpr int kMaxRestartsPerWindow = 5;
  static constexpr int kRestartWindowSeconds = 60;
};
```

#### `python_be.cc` — `RestartStubProcess()`

Key design decisions:
1. **Do NOT call `UpdateHealth()` or `IsHealthy()`** — the stub is dead
2. **Kill before terminating monitor** — `TerminateStub()` sends SIGKILL to handle zombies
3. **Rate limit** — track restarts per time window
4. **No request retry** — fail the current request, restart for next

#### `python_be.cc` — `TRITONBACKEND_ModelInstanceExecute()`

Key design decisions:
1. **Detect via `IsStubProcessAlive()`**, not string matching
2. **Guard with mutex** to prevent concurrent restart attempts
3. **Do NOT retry the current request** — return error for this batch
4. **Log clearly** for observability

#### `python_be.cc` — Remove TODO comment

The comment `// TODO: Implement restart on decoupled` should be removed and replaced with the actual implementation.

### Test Plan for PR-A

#### Integration Test: `test_stub_restart.sh`

```bash
#!/bin/bash
# Test that stub restart recovers from unhealthy state

MODEL_NAME="restart_test"
SERVER_ARGS="--model-repository=${MODELDIR} --backend-directory=${BACKEND_DIR} --log-verbose=1"

# Setup model that kills stub on input=0
mkdir -p models/${MODEL_NAME}/1/
cat > models/${MODEL_NAME}/1/model.py << 'EOF'
import triton_python_backend_utils as pb_utils
import numpy as np
import os

class TritonPythonModel:
    def initialize(self, args):
        pass
    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()
            if inp[0] == 0:
                os._exit(0)
            out = pb_utils.Tensor("OUTPUT", inp + 1)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
EOF

cat > models/${MODEL_NAME}/config.pbtxt << EOF
name: "${MODEL_NAME}"
backend: "python"
max_batch_size: 0
input [{ name: "INPUT" data_type: TYPE_INT32 dims: [1] }]
output [{ name: "OUTPUT" data_type: TYPE_INT32 dims: [1] }]
EOF

# Start server
run_server

# Test 1: Normal inference succeeds
python3 test_stub_restart.py TestStubRestart.test_normal_inference

# Test 2: Kill stub, verify restart, verify next inference succeeds
python3 test_stub_restart.py TestStubRestart.test_restart_after_kill

# Test 3: Model readiness recovers after restart
python3 test_stub_restart.py TestStubRestart.test_readiness_after_restart

# Test 4: Rate limiting (kill stub repeatedly, verify it stops restarting)
python3 test_stub_restart.py TestStubRestart.test_restart_rate_limit

kill_server
```

#### Python Test: `test_stub_restart.py`

```python
import unittest
import time
import numpy as np
import tritonclient.http as httpclient

class TestStubRestart(unittest.TestCase):
    def setUp(self):
        self.client = httpclient.InferenceServerClient("localhost:8000")
        self.model_name = "restart_test"

    def _infer(self, value):
        inp = httpclient.InferInput("INPUT", [1], "INT32")
        inp.set_data_from_numpy(np.array([value], dtype=np.int32))
        return self.client.infer(self.model_name, [inp])

    def test_normal_inference(self):
        result = self._infer(1)
        output = result.as_numpy("OUTPUT")
        self.assertEqual(output[0], 2)

    def test_restart_after_kill(self):
        # Normal inference succeeds
        self._infer(1)

        # Kill stub
        with self.assertRaises(Exception):
            self._infer(0)

        # Wait for restart
        time.sleep(3)

        # Next inference should succeed after restart
        result = self._infer(42)
        output = result.as_numpy("OUTPUT")
        self.assertEqual(output[0], 43)

    def test_readiness_after_restart(self):
        # Kill stub
        with self.assertRaises(Exception):
            self._infer(0)

        # Wait for restart
        time.sleep(3)

        # Model should be ready again
        self.assertTrue(self.client.is_model_ready(self.model_name))

    def test_restart_rate_limit(self):
        # Kill stub repeatedly (more than kMaxRestartsPerWindow)
        for i in range(7):
            try:
                self._infer(0)
            except Exception:
                pass
            time.sleep(1)

        # After exceeding rate limit, model should stay unhealthy
        self.assertFalse(self.client.is_model_ready(self.model_name))
```

---

## PR-B: Server Core — Runtime Readiness Check

### Target

Open new PR in `triton-inference-server/core`.

### Changed Files

| File | Changes |
|------|---------|
| `src/server.h` | Add `runtime_readiness_check_` flag |
| `src/server.cc` | Modify `IsReady()` to call `ModelIsReady()` per-model |
| `include/triton/core/tritonserver.h` | Add `TRITONSERVER_ServerOptionsSetRuntimeReadinessCheck()` |
| `src/tritonserver.cc` | Implement the new option setter |

### Implementation Summary

When `strict_readiness_=true` AND `runtime_readiness_check_=true`:
- `IsReady()` iterates all models with `READY` lifecycle state
- For each, calls `ModelIsReady()` which invokes per-instance backend checks
- If any model instance reports unhealthy, server readiness returns false

### Rollout Strategy

1. **Phase 1**: Add behind `TRITON_ENABLE_RUNTIME_READINESS_CHECK` env var (default: OFF)
2. **Phase 2**: After validation in production, change default to ON for `strict_readiness_=true`
3. **Phase 3**: Remove the flag, make it always-on when strict readiness is enabled

### Test Plan for PR-B

```bash
# Test: server-level readiness reflects stub death
# 1. Start server with --strict-readiness=true and TRITON_ENABLE_RUNTIME_READINESS_CHECK=1
# 2. Verify /v2/health/ready returns 200
# 3. Kill Python backend stub (via signal or os._exit)
# 4. Verify /v2/health/ready returns non-200
# 5. (If restart is enabled) Wait for restart, verify /v2/health/ready returns 200 again
```

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Submit improved PR-A (or review comments on PR #431) |
| 1 | Run CI on PR-A, fix any build/test failures |
| 2 | Submit PR-B to triton core |
| 2-3 | Address code review feedback on both PRs |
| 3-4 | Merge PR-A, then PR-B |
| 4+ | Backport to release branches if requested |

## Open Questions for Maintainers

1. Should restart be enabled by default, or behind a config flag?
2. What is the acceptable performance overhead for runtime readiness checks in `IsReady()`?
3. Should decoupled model restart be blocked until properly implemented, or should it be attempted with best-effort cleanup?
4. Is `kMaxRestartsPerWindow = 5` a reasonable default, or should it be configurable via backend config?
