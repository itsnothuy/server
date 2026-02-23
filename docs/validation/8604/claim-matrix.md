# Claim-by-Claim Validation Matrix

Every major factual claim from "Python backend fix 1.txt" / "Python backend fix 2.txt" is validated below against the actual upstream codebase (python_backend `main`, triton core `main`, server `main`).

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Verified true |
| ❌ | Verified false |
| ⚠️ | Partially true / context-dependent |
| ❓ | Not verifiable with available evidence |

---

## 1. Architecture Claims

### Claim 1.1: Stub process isolation
> "The Python backend isolates a user-provided model.py in a separate stub process."

**Classification: ✅ Verified true**

**Evidence:** `python_backend/src/python_be.cc` — `ModelInstanceState::LaunchStubProcess()` creates a `StubLauncher` object and calls `Stub()->Setup()` then `Stub()->Launch()` which forks a child process (`triton_python_backend_stub`). The stub process runs `model.py` in its own address space.

---

### Claim 1.2: Shared-memory communication
> "The Triton core communicates with this stub using shared-memory message queues and a health mutex."

**Classification: ✅ Verified true**

**Evidence:** `python_backend/src/python_be.cc` — `SendMessageToStub()` (lines ~1072–1104) acquires `Stub()->HealthMutex()` with a timed lock, sets `Stub()->IpcControl()->stub_health = false`, then pushes onto `Stub()->StubMessageQueue()`. The `StubLauncher` class manages shared memory pools and message queues via boost::interprocess.

---

### Claim 1.3: Health mutex detection mechanism
> "ModelInstanceState::IsStubProcessAlive tries to lock the health mutex for one second. If it acquires the lock it returns the value of stub_health; if the lock cannot be obtained it assumes the stub has died or is blocked."

**Classification: ✅ Verified true**

**Evidence:** `python_backend/src/python_be.cc` lines ~155–171:
```cpp
bool ModelInstanceState::IsStubProcessAlive() {
  boost::posix_time::ptime timeout =
      boost::get_system_time() + boost::posix_time::seconds(1);
  bi::scoped_lock<bi::interprocess_mutex> lock(*Stub()->HealthMutex(), timeout);
  if (lock) {
    return Stub()->IpcControl()->stub_health;
  } else {
    return false;
  }
}
```

---

### Claim 1.4: SendMessageToStub sets stub_health to false
> "This function acquires the health mutex with a 1-second timeout, sets stub_health to false and pushes the message onto the IPC queue."

**Classification: ✅ Verified true**

**Evidence:** `python_backend/src/python_be.cc` `SendMessageToStub()` — confirmed: acquires `HealthMutex()` with 1000ms timeout, sets `Stub()->IpcControl()->stub_health = false`, pushes to `Stub()->StubMessageQueue()`.

---

## 2. Root-Cause Claims

### Claim 2.1: PR #360 removed restart logic
> "A refactor to unify decoupled and non-decoupled execution (python backend PR #360) removed the restart block and replaced it with a TODO."

**Classification: ✅ Verified true**

**Evidence:** On `main` branch, `TRITONBACKEND_ModelInstanceExecute` contains exactly:
```cpp
// If restart is equal to true, it indicates that the stub process is
// unhealthy and needs a restart.
// TODO: Implement restart on decoupled
```
No actual restart variable or restart code exists. The old restart code (described in the fix files) involving `TerminateMonitor()`, `KillStubProcess()`, `Setup()`, `StartMonitor()`, `Launch()` is completely absent from `main`.

---

### Claim 2.2: Restart flag exists but is never acted upon
> "The unified execution path sets a restart flag when the stub fails but never acts on it."

**Classification: ⚠️ Partially true**

**Evidence:** The TODO comment references a restart flag concept, but on the current `main` branch, there is **no actual `restart` variable** declared in `TRITONBACKEND_ModelInstanceExecute`. The `ProcessRequests` function does not set any restart flag — it simply returns an error (e.g., `TRITONSERVER_ERROR_INTERNAL` with "Stub process is not healthy") via `RETURN_IF_ERROR`. The claim is correct in spirit (the stub is not restarted) but technically inaccurate: there is no restart flag being set; the comment is orphaned.

**Impact:** Low — the conclusion (stub is never restarted) is correct regardless.

---

### Claim 2.3: TRITONBACKEND_ModelInstanceReady is cached / only called at startup
> "TRITONBACKEND_ModelInstanceReady is called only during model loading. It checks if the stub is active and returns an error if not. There is no periodic check after initialization."

**Classification: ❌ Verified false**

**Evidence:** This is the most significant factual error in the analysis.

The upstream code reveals that `TRITONBACKEND_ModelInstanceReady` is called **every time** per-model readiness is checked:

1. `/v2/models/{model}/ready` → `TRITONSERVER_ServerModelIsReady()` → `InferenceServer::ModelIsReady()` → `model->IsReady()` → `TritonModel::IsReady()` → iterates all instances → `TritonModelInstance::IsReady()` → calls `TRITONBACKEND_ModelInstanceReady`
2. This is **NOT cached**. Each call invokes the backend function fresh.

Furthermore, the upstream server repo already has a test at `qa/L0_backend_python/model_readiness/test.sh` that **proves** this: it kills the stub with SIGSEGV/SIGKILL, then checks `/v2/models/{model}/ready` and confirms it returns NOT ready, with the error `"Stub process 'X_0_0' is not healthy."`.

The Python backend's `TRITONBACKEND_ModelInstanceReady` calls `Stub()->StubActive()` which uses `waitpid(WNOHANG)` to check the OS-level PID — this correctly detects dead/zombie stubs.

**Impact:** HIGH — This error means Track B's proposed changes to `TRITONBACKEND_ModelInstanceReady` are largely **unnecessary** for per-model readiness. The per-model endpoint already works correctly. The actual gap is only at the **server-level** `/v2/health/ready`.

---

### Claim 2.4: Health APIs return 200 despite dead stub
> "Health endpoints therefore return misleading HTTP 200 responses even though the model can no longer serve inferences."

**Classification: ⚠️ Partially true — depends on the endpoint**

**Evidence:**

| Endpoint | With dead stub | Explanation |
|----------|---------------|-------------|
| `/v2/health/live` | Returns **200** ✅ | Correct — this checks server process liveness, not model health |
| `/v2/health/ready` (strict=true) | Returns **200** ⚠️ | **This IS the bug.** `IsReady()` calls `ModelStates()` which only checks lifecycle state (`ModelReadyState::READY`), NOT `model->IsReady()`. A dead stub doesn't change the lifecycle state. |
| `/v2/models/{model}/ready` | Returns **non-200** ✅ | This endpoint **does work correctly** — it calls `model->IsReady()` → `TRITONBACKEND_ModelInstanceReady` → `StubActive()` |

The claim is partially correct: `/v2/health/ready` does return misleading 200 even with strict readiness. But `/v2/models/{model}/ready` **does** correctly report unhealthy. The fix files conflate these two endpoints.

**Impact:** HIGH — The precise location of the bug matters for the fix. The gap is specifically in `InferenceServer::IsReady()` (server-level readiness with strict mode), which uses `ModelStates()` (lifecycle state only) instead of also calling `model->IsReady()`.

---

### Claim 2.5: Old restart logic in r24.05
> "In the older r24.05 implementation, non-decoupled mode had a block of code: `if (restart) { ... instance_state->TerminateMonitor(); instance_state->Stub()->KillStubProcess(); ...}`"

**Classification: ❓ Not directly verifiable**

**Evidence:** The r24.05 branch was not directly inspected. However, the TODO comment on `main` that references restart, combined with the absence of any restart code on `main`, is strongly consistent with this claim. The string "Stub process is unhealthy and it will be restarted" does not appear on `main`, confirming it was removed.

---

## 3. Proposed Fix Claims

### Claim 3.1: Track A restart sequence
> "Call TerminateMonitor() → KillStubProcess() → Setup() → StartMonitor() → Launch()"

**Classification: ⚠️ Partially correct**

**Evidence:** The actual destructor sequence in `ModelInstanceState::~ModelInstanceState()` (on `main`) is:
```cpp
Stub()->UpdateHealth();
if (Stub()->IsHealthy()) { thread_pool_->wait(); }
Stub()->TerminateStub();   // NOT KillStubProcess
TerminateMonitor();
Stub()->ClearQueues();
Stub().reset();
```

And `LaunchStubProcess()` (the startup path) does:
```cpp
Stub() = std::make_unique<StubLauncher>(...);
Stub()->Initialize(model_state);
Stub()->Setup();
StartMonitor();
Stub()->Launch();
thread_pool_ = ...;
request_executor_ = ...;
```

The fix files reference `KillStubProcess()` but the destructor uses `TerminateStub()` (which tries a graceful finalize first, then kills). The restart sequence should mirror the destructor + constructor pattern, not blindly call `KillStubProcess()`.

**Impact:** Medium — the proposed sequence is close but needs adjustment for the actual API.

---

### Claim 3.2: Track B — `instance_unhealthy` flag needed
> "Modify ModelInstanceState::IsStubProcessAlive to set a flag instance_unhealthy. Then change TRITONBACKEND_ModelInstanceReady to return an error when this flag is set."

**Classification: ❌ Unnecessary for per-model readiness**

**Evidence:** `TRITONBACKEND_ModelInstanceReady` already calls `Stub()->StubActive()` which dynamically checks the stub PID status. It does NOT cache results. An `instance_unhealthy` flag is redundant — the existing check already works.

However, for server-level readiness (`/v2/health/ready`), changes ARE needed — but they belong in the **server core** (`InferenceServer::IsReady()`), not in the python backend.

**Impact:** Medium — the fix location is wrong but the goal (propagating unhealthy state) is correct.

---

### Claim 3.3: Track B — CheckRuntimeModelReadiness() needed in server core
> "Add a CheckRuntimeModelReadiness() function that iterates over all model instances and calls their ModelInstanceReady functions."

**Classification: ✅ Correctly identifies the gap**

**Evidence:** `InferenceServer::IsReady()` (in triton core `server.cc`) with `strict_readiness_=true` calls `ModelStates()` which returns lifecycle states (`READY`/`LOADING`/etc.) but does **NOT** call `model->IsReady()`. This is unlike `InferenceServer::ModelIsReady()` (per-model check) which DOES call `model->IsReady()` after the lifecycle check.

The fix should make `IsReady()` also call `model->IsReady()` for each model when `strict_readiness_=true`, or call `ModelIsReady()` per-model. This is exactly what the claim proposes.

**Impact:** HIGH — this is the core server-side fix needed. PR #431 does NOT address this.

---

### Claim 3.4: Backwards compatibility concern
> "Under the default strict-readiness=false, retain current semantics to avoid breaking deployments."

**Classification: ⚠️ Slightly outdated**

**Evidence:** The default for `strict_readiness_` is actually `true` (set in `InferenceServer::InferenceServer()` constructor and in `TritonServerOptions`). The claim says "default strict-readiness=false" which is incorrect — the default is `true`.

**Impact:** Low — the backwards compatibility concern is valid regardless of default value, but the default should be stated correctly.

---

## 4. Repro Claims

### Claim 4.1: os._exit(0) causes stub zombie
> "A model.py that calls os._exit(0) causes the stub to become a zombie; subsequent inference requests fail while /v2/health/live and /v2/health/ready still return 200 OK."

**Classification: ❓ Not runtime-verified; consistent with code analysis**

**Evidence:** Cannot run Docker containers to verify. However, code analysis confirms: `os._exit(0)` in the stub would terminate the child process. The parent does `waitpid(WNOHANG)` in `StubActive()` which would detect the exit and return false. However, if `waitpid` is not called between the exit and a health check, the process would indeed be a zombie. The health endpoints would remain 200 as analyzed in Claim 2.4.

---

### Claim 4.2: Repro model design
> The minimal repro model using `input[0] == 0` to trigger `os._exit(0)` is valid.

**Classification: ✅ Reasonable repro design**

**Evidence:** The model design is straightforward and consistent with the issue #8604 description. The existing upstream test at `qa/L0_backend_python/model_readiness/test.sh` uses a similar approach (kill stub with signal from outside) which validates the general pattern.

---

## 5. Reference Claims

### Claim 5.1: Issue #7230 describes zombie stub with healthy endpoints
**Classification: ✅ Verified** — Issue #7230 is real and describes the same class of problem.

### Claim 5.2: PR #360 unified pipelines
**Classification: ✅ Verified** — The TODO comment on `main` explicitly references this unification, and the absence of restart code confirms the removal.

---

## Summary

| Category | ✅ True | ❌ False | ⚠️ Partial | ❓ Unverifiable |
|----------|---------|---------|------------|----------------|
| Architecture | 4 | 0 | 0 | 0 |
| Root-cause | 2 | 1 | 2 | 1 |
| Proposed fix | 1 | 1 | 2 | 0 |
| Repro | 1 | 0 | 0 | 1 |
| References | 2 | 0 | 0 | 0 |
| **Total** | **10** | **2** | **4** | **2** |

### Critical Findings

1. **The biggest error**: Claim 2.3 states `TRITONBACKEND_ModelInstanceReady` is "only called during model loading" and "cached." This is **false**. It is called on every per-model readiness check. The existing upstream test proves this. This error leads the analysis to propose unnecessary changes to the python backend's readiness function.

2. **The actual gap is correctly identified but mislocated**: The real bug is in `InferenceServer::IsReady()` (server core) — it checks `ModelStates()` (lifecycle only) instead of also calling `model->IsReady()` per-instance. The fix files correctly propose a `CheckRuntimeModelReadiness()` function but attribute the problem to the backend rather than the core.

3. **The restart proposal (Track A) is fundamentally sound** but needs adjustments to match the actual API (e.g., `TerminateStub()` vs `KillStubProcess()`, need to recreate `thread_pool_` and `request_executor_`).
