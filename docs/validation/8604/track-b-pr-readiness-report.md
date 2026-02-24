# Track B PR-Readiness Validation Report

> **Reviewer model:** Claude Opus 4.6  
> **Date:** 2025-02-24  
> **Patch under review:** `fix-8604/core/src/server.cc.patch.cpp`  
> **Target upstream:** `triton-inference-server/core`, function `InferenceServer::IsReady()`  
> **Evidence base:** All upstream source code gathered via `github_repo` searches against `triton-inference-server/core` `main` branch

---

## Section 1: Code Correctness

### Q1: Are the argument types to `ModelIsReady(mv.first.name_, vs.first, &model_ready)` correct?

**YES — all three arguments are type-correct.**

Evidence chain:

- `ModelStates()` returns `const ModelStateMap` ([model_repository_manager.cc L1126](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_repository_manager.cc#L1126)).
- `ModelStateMap = std::map<ModelIdentifier, VersionStateMap>` ([model_lifecycle.h L96](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_lifecycle.h#L96)).
- Therefore `mv.first` is `ModelIdentifier`.
- `ModelIdentifier` is a struct with public member `std::string name_` ([model.h L91-94](https://github.com/triton-inference-server/core/tree/main/src/model.h#L91-L94)). So `mv.first.name_` is `std::string`.
- `VersionStateMap = std::map<int64_t, std::pair<ModelReadyState, std::string>>` ([model_lifecycle.h L86](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_lifecycle.h#L86)). So `vs.first` is `int64_t`.
- `ModelIsReady` signature: `Status ModelIsReady(const std::string& model_name, const int64_t model_version, bool* ready)` ([server.cc L459-461](https://github.com/triton-inference-server/core/tree/main/src/server.cc#L459-L461)).
- Match: `std::string` → `const std::string&` ✓, `int64_t` → `const int64_t` ✓, `bool*` → `bool*` ✓.

**However — there is a subtle semantic issue with namespaces:** `ModelIsReady()` internally constructs `ModelIdentifier("", model_name)` via `GetModel()` → `ModelRepositoryManager::GetModel(const std::string&, ...)` → `find_identifier_fn_()` ([model_repository_manager.cc L1217-1222](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_repository_manager.cc#L1217-L1222)). This means it reconstructs the identifier from just the name, discarding the namespace. When model namespacing is enabled (`enable_model_namespacing_=true`), `find_identifier_fn_` does a global lookup. When namespacing is disabled (the default), it's a no-op that always returns success with the empty namespace. **This is safe for the default case** and matches how `ModelIsReady()` is already called from external endpoints. But it could fail to find the correct model if two models share the same name in different namespaces. **This is an existing limitation of `ModelIsReady()` itself, not introduced by this patch.**

### Q2: Does the `goto strict_done` cross any variable initializations?

**NO — no compile error.**

The variables `bool model_ready` and `Status status` are declared inside the `if (vs.second.first == ModelReadyState::READY) { ... }` block. The `goto strict_done` jumps OUT of this block (and out of both `for` loops) to `strict_done:` which is after the loops. In C++, jumping *out of* a scope that contains variable declarations is always legal — the variables are destroyed. The "jump bypasses variable initialization" error only occurs when a `goto` jumps *into* a scope that contains declarations with initializers, skipping the declaration. Here the jump is outward, so there is no issue.

Additionally, the existing code already uses `goto strict_done` in the same loop structure with similar patterns. The original code compiles, and the new code follows the same pattern.

### Q3: Is the `if (vs.second.first == ModelReadyState::READY)` logic correct? Is there an "unloaded" edge case?

**THIS IS THE MOST CRITICAL FINDING. There is a potential behavioral regression.**

The logic flow for the "unloaded" edge case:

1. Original existing check: `if ((vs.second.first != ModelReadyState::READY) && (vs.second.second != "unloaded"))`
2. If state IS `READY`: the first clause `(state != READY)` is `false`, so the entire AND is `false` regardless of reason. The `goto` is NOT taken. Falls through.
3. If state is NOT `READY` AND reason IS `"unloaded"`: the second clause is `false`, so the AND is `false`. The `goto` is NOT taken. Falls through. The comment says "Okay if model is not ready due to unload."
4. The new block fires only when `vs.second.first == ModelReadyState::READY`.

**Analysis of what happens for each case:**

- **Case A: state=`READY`, reason="" (normal healthy model):** New block fires. `ModelIsReady()` is called. Expected: returns `true`. Server reports ready. ✓ Correct.

- **Case B: state=`UNAVAILABLE`, reason=`"unloaded"`:** Existing check: `(true) && (false)` = `false`. Falls through. New block: `UNAVAILABLE != READY`, so new block does NOT fire. Falls through to next iteration. ✓ Correct — unloaded models are still accepted.

- **Case C: state=`READY`, reason=`"unloaded"`:** Can this actually happen? Looking at the lifecycle code: when a model is unloaded, `ModelInfo::Release()` sets `state_ = ModelReadyState::UNLOADING` ([model_lifecycle.h L252-257](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_lifecycle.h#L252-L257)). The state transitions through `UNLOADING` → `UNAVAILABLE`. A model should never be in state=`READY` with reason=`"unloaded"`. **However**, I cannot rule out all possible transient states or race conditions. The original code was defensive about this — it treated `(READY, "unloaded")` as acceptable. The new code would call `ModelIsReady()` for this combination. What would happen?
  - `ModelIsReady()` calls `GetModel()` ([server.cc L475](https://github.com/triton-inference-server/core/tree/main/src/server.cc#L475)). For an unloading model, `GetModel()` acquires `map_mtx_` and checks if state is READY ([model_lifecycle.cc L377-388](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_lifecycle.cc#L377-L388)). If the model is truly being unloaded, the state in the lifecycle map would have transitioned away from READY. `GetModel()` would return a NOT_FOUND or UNAVAILABLE error. `ModelIsReady()` would set `*ready = false` and return `Status::Success`. The server would report not-ready.
  - **This is arguably the correct behavior** — if a model is in a state where `GetModel()` can't serve it, reporting the server as not-ready is safer than reporting it as ready.

**Verdict: The "unloaded" edge case is NOT a real problem in practice** because `(READY, "unloaded")` is not a valid state transition. The existing code's defensiveness against it is preserved for the `UNAVAILABLE/unloaded` case. The `READY` case would only fire if the model is actually in READY lifecycle state, at which point checking runtime readiness is correct.

### Q4: Does `ModelIsReady()` hold any locks that `IsReady()` also holds? (Deadlock risk)

**NO — no deadlock.**

- `IsReady()` does not acquire any mutex directly. It calls `ModelStates()` which acquires `map_mtx_`, copies the entire map, releases `map_mtx_`, and returns the copy ([model_lifecycle.cc L275-293](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_lifecycle.cc#L275-L293)). By the time the new code calls `ModelIsReady()`, the lock is already released.
- `ModelIsReady()` → `GetModel()` → `ModelRepositoryManager::GetModel(string)` → `find_identifier_fn_()` then `ModelLifeCycle::GetModel()` which acquires `map_mtx_` independently ([model_lifecycle.cc L339-341](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_lifecycle.cc#L339-L341)).
- `ModelIsReady()` also calls `ModelState()` which acquires `map_mtx_` independently ([model_lifecycle.cc L312-314](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_lifecycle.cc#L312-L314)).
- Since `ModelStates()` returns a copy and releases the lock, there is no held-lock overlap. No deadlock.

### Q5: Does `ModelIsReady()` have its own `ScopedAtomicIncrement`?

**YES — this is a double-increment issue, but it is benign.**

- `IsReady()` has `ScopedAtomicIncrement inflight(inflight_request_counter_)` at its top ([server.cc L429](https://github.com/triton-inference-server/core/tree/main/src/server.cc#L429)).
- `ModelIsReady()` has its own `ScopedAtomicIncrement inflight(inflight_request_counter_)` ([server.cc L471](https://github.com/triton-inference-server/core/tree/main/src/server.cc#L471)).
- When `IsReady()` calls `ModelIsReady()`, the counter is incremented twice.
- `InferenceServer::Stop()` waits for `inflight_request_counter_ == 0` before completing shutdown ([server.cc L347-368](https://github.com/triton-inference-server/core/tree/main/src/server.cc#L347-L368)).
- **Impact:** During a health probe, the counter is 2 instead of 1. This could delay shutdown by a negligible amount (the health probe call completes in microseconds). This is **not a correctness bug** — `Stop()` polls and waits, so the counter will reach 0 once the health probe completes.
- `ModelIsReady()` also checks `ready_state_ != SERVER_READY` and returns UNAVAILABLE if the server is not ready. Since we're inside `IsReady()` which already verified `ready_state_ == SERVER_READY`, and `ModelIsReady()` is called within the `ScopedAtomicIncrement` scope (so the server can't transition to not-ready during this check due to the inflight counter), this is safe.

**Section 1 Verdict: READY (with one minor note about double inflight increment — cosmetic, not a bug)**

---

## Section 2: Compilation Readiness

### Q1: Are all symbols available at the point of insertion?

| Symbol | Available? | Evidence |
|--------|-----------|----------|
| `ModelIsReady` | ✅ | Member function of `InferenceServer`, declared in server.h, defined in server.cc L459 |
| `mv.first.name_` | ✅ | `ModelIdentifier::name_` is a public `std::string` member (model.h L91) |
| `LOG_VERBOSE(1)` | ✅ | Used elsewhere in server.cc (e.g., Stop() at L347-368, ModelIsReady() at L484) |
| `ModelReadyState::READY` | ✅ | Enum used in the original `IsReady()` code at L440 |
| `Status` | ✅ | Used everywhere in server.cc |
| `bool` | ✅ | Trivially available |
| `vs.first` | ✅ | `int64_t` from `VersionStateMap` |

All symbols are available. No new `#include` is needed.

### Q2: Is `LOG_VERBOSE` the correct macro?

**YES.** `LOG_VERBOSE(1)` is used directly in `ModelIsReady()` in the same file at L484-486:
```cpp
LOG_VERBOSE(1) << "Model '" << model_name << "' version "
               << model->Version()
               << " is not ready: " << status.Message();
```
The patch follows the same pattern and verbosity level. This is correct.

### Q3: Does the `goto` inside a nested `if` compile?

**YES.** The `goto strict_done` is used to jump out of nested loops — this is standard C++ and is the same pattern used in the original code at L447-448. The variables `bool model_ready` and `Status status` are declared inside the `if` block. When `goto` jumps out, these variables go out of scope and are destroyed. No "jump bypasses initialization" error occurs because:
- The `goto` jumps forward to `strict_done:` which is at a higher scope level.
- The variables are not in scope at the label.
- C++ only errors when a `goto` jumps *past* a declaration *into* the scope where that declaration is visible from the goto target. Here, the jump is *out of* the scope.

**Section 2 Verdict: READY**

---

## Section 3: Behavioral Correctness at the Boundaries

### Scenario 1: All models healthy

`ModelStates()` returns all models with `ModelReadyState::READY`. The existing check passes (state is READY, so the `!=READY` clause is false, no goto). The new block fires: `ModelIsReady()` returns `true` for each. Server reports ready (200).

**Preserved. ✓**

### Scenario 2: One model loading (state = `LOADING`)

`vs.second.first` is `LOADING`. The existing check: `(LOADING != READY)` = `true`, `(reason != "unloaded")` = `true` (reason would be something like "loading"). AND is `true`. `goto strict_done` fires. `*ready = false`. Server reports 503.

The new `if (vs.second.first == READY)` is never reached.

**Preserved. ✓**

### Scenario 3: One model unloaded (state = `UNAVAILABLE`, reason = `"unloaded"`)

Existing check: `(UNAVAILABLE != READY)` = `true`, `("unloaded" != "unloaded")` = `false`. AND is `false`. Falls through. New block: `UNAVAILABLE != READY`, so `if (READY)` does NOT fire. Falls through. Model is skipped. Server reports ready.

**Preserved. ✓**

### Scenario 4: Server has zero models loaded

`model_versions` is empty. The `for` loop doesn't execute. `*ready` remains `true`. No new code is reached. Server reports 200.

**Preserved. ✓**

### Scenario 5: `ModelIsReady()` returns a non-OK `Status`

Looking at the implementation: `ModelIsReady()` ([server.cc L459-487](https://github.com/triton-inference-server/core/tree/main/src/server.cc#L459-L487)) checks `ready_state_` first and returns `Status(UNAVAILABLE, "Server not ready")` if the server is not in READY state. Since we're inside `IsReady()` which already verified `ready_state_ == SERVER_READY`, and the inflight counter prevents state transitions, this path won't fire.

After that, `ModelIsReady()` catches all errors from `GetModel()` and `ModelState()` silently (uses `.IsOk()` checks, not `RETURN_IF_ERROR`). If `GetModel()` fails, `*ready` stays `false` and `Status::Success` is returned. If `ModelState()` fails, same behavior.

**The only case where `ModelIsReady()` returns a non-OK Status is when `ready_state_ != SERVER_READY`.** This won't happen in our call path. The patch's `!status.IsOk()` check is belt-and-suspenders — correct defensive coding. It treats server-not-ready as "not ready" which is correct.

**Preserved / correct. ✓**

**Section 3 Verdict: READY**

---

## Section 4: Performance Analysis

### Q1: Call frequency scalability

With N models and K instances per model, each health probe triggers:
- 1 `ModelStates()` call (copies entire map, acquires `map_mtx_` once)
- Up to N × V calls to `ModelIsReady()` (N models × V versions per model)

Each `ModelIsReady()` call:
1. `ScopedAtomicIncrement` — atomic increment/decrement (nanoseconds)
2. `GetModel()` — acquires `map_mtx_`, looks up model (microseconds)
3. `ModelState()` — acquires `map_mtx_` again (microseconds)
4. `model->IsReady()` — iterates all instances, calls `TRITONBACKEND_ModelInstanceReady` per instance

For Python backend: `StubActive()` → `waitpid(WNOHANG)` — ~1-2µs per instance.
For non-Python backends: null function pointer check → return Success — effectively free.

**Worst case:** 1000 models × 1 version × 4 instances = 4000 `waitpid` calls + 2000 `map_mtx_` lock acquisitions. At ~2µs each, total ~12ms. At 1s probe interval, this is 1.2% overhead. At 10s probe interval, 0.12%.

**Assessment:** The overhead is measurable but negligible for realistic deployments. A deployment with 1000+ models and 1s health probes would already have other bottlenecks. The `map_mtx_` contention is the bigger concern — each `ModelIsReady()` call acquires the mutex twice (`GetModel()` and `ModelState()`). With many models, this could briefly block inference-path operations that also need the mutex. However, the lock is held only for map lookups (microseconds), not for inference execution.

**The performance claim is valid.**

### Q2: Lock contention with inference threads

`ModelIsReady()` → `GetModel()` → `ModelLifeCycle::GetModel()` acquires `map_mtx_` ([model_lifecycle.cc L341](https://github.com/triton-inference-server/core/tree/main/src/model_repository_manager/model_lifecycle.cc#L341)). This is the same mutex used by:
- `ModelStates()` — already called from original `IsReady()`
- Inference request routing (when Triton looks up a model to serve a request)

The new code adds N additional lock acquisitions (2 per model: GetModel + ModelState). In the original code there was 1 lock acquisition (ModelStates). The increase is significant in relative terms (1 → 2N+1) but each acquisition holds the lock for microseconds (just a map lookup).

**Not a blocking concern, but worth mentioning in the PR for maintainer awareness.**

### Q3: `inflight_request_counter_` during shutdown

As analyzed in Section 1 Q5, the counter is incremented twice. `InferenceServer::Stop()` polls until the counter reaches 0. The double increment extends the "apparent inflight" window by the duration of the `ModelIsReady()` calls (microseconds). This has no practical impact — shutdown is already designed to wait for in-flight operations.

**Section 4 Verdict: READY**

---

## Section 5: Test Quality

### Q1: Test 1 — does `run_server` wait for readiness?

**NEEDS VERIFICATION.** The `run_server` function is sourced from `../common/util.sh`. In the standard Triton QA framework, `run_server` starts the server in the background and polls `/v2/health/ready` until it returns 200 or a timeout expires. **This is almost certainly the case** — every other QA test uses `run_server` this way. But this test has not been run against the actual QA infrastructure.

### Q2: Test 2 — is `sleep 2` necessary?

**`sleep 2` is unnecessary and should be removed.** The Python test already has a 10-second polling loop with 0.5s intervals. After `kill -9`, the process is dead immediately. The next call to `IsStubProcessAlive()` will return `false`. The `sleep 2` adds latency without preventing any race condition that the Python polling loop doesn't already handle.

**Recommendation:** Remove `sleep 2` from both Test 2 and Test 4.

### Q3: Test 3 — risk of prior state contamination

**Low risk but present.** After the stub is killed in Test 2, the Triton server is still running. The stub death does not crash the server — it simply makes the model unservable. Test 3 checks per-model readiness, which should correctly report the model as not-ready. There is no mechanism by which the stub death would cause the server to enter an error state that prevents health endpoint responses.

**The test ordering dependency is intentional and documented (Tests 2 and 3 run sequentially after the same stub kill).** This is acceptable.

### Q4: Test 4 — is there a wait after restart before killing stub?

**`run_server` handles this.** As noted in Q1, `run_server` blocks until the server is ready (model loaded). After `run_server` returns, the model is loaded and the stub is running. The subsequent `pgrep` and `kill` are safe.

### Q5: Is `pgrep -f` reliable?

**NOT FULLY RELIABLE.** The pattern `"triton_python_backend_stub.*identity_fp32"` depends on:
- The exact process name (`triton_python_backend_stub`) — this could change across versions.
- The model name appearing in the command-line arguments — this is an implementation detail of the Python backend.

The test does handle the empty `$STUB_PID` case with an error message, which is good. However, `pgrep -f` could also match multiple processes (e.g., if the stub spawns child processes, or if another model's stub name contains "identity_fp32" as a substring).

**This is consistent with how other Triton QA tests find stub processes** (e.g., `qa/L0_backend_python/` uses similar `pgrep` patterns). It's not ideal but follows project convention.

### Q6: Does the test use `requests` or should it use `tritonclient`?

**This is a DEFECT.** The existing Triton QA tests under `qa/L0_backend_python/` consistently use `tritonclient.http` or `tritonclient.grpc`, not the bare `requests` library. While `requests` is typically available in the Triton container, using `tritonclient` would:
1. Be consistent with the rest of the test suite.
2. Use the project's own client library as intended.
3. Avoid a dependency assumption.

However, for health-check-only tests (no inference calls), some tests do use `requests` or `urllib`. Looking at the purpose — the test only calls `/v2/health/ready` and `/v2/models/.../ready`, both of which are simple GET requests returning a status code. `tritonclient.http.InferenceServerClient.is_server_ready()` and `.is_model_ready()` could replace the raw HTTP calls.

**Recommendation:** Replace `import requests` with `tritonclient.http` for consistency, or at minimum verify that `requests` is available in the CI container. This is a soft defect — it won't cause functional issues if `requests` is installed, but it deviates from project convention.

### Additional test issue: Copyright year

The test files use `Copyright 2026` — this is a typo. Should be `2025`.

**Section 5 Verdict: NOT READY — needs `sleep 2` removal, copyright year fix, and preferably `tritonclient` migration**

---

## Section 6: New Contributor Checklist Audit

### 1. CONTRIBUTING.md / Code of Conduct
**NEEDS VERIFICATION.** The `triton-inference-server/core` repo has a `CONTRIBUTING.md` in the `server` repo. The contributor should read it and confirm compliance. The commit message format and test structure should be verified against upstream conventions.

### 2. Drive-by / random changes
**PASS.** This is a genuine, clearly motivated bug fix for a specific reported issue (#8604). The change is minimal (one new code block in one function). No cosmetic or unrelated changes.

### 3. Opening PR without checking if project wants change
**NEEDS VERIFICATION.** There is no evidence that Triton maintainers have committed to fixing this differently or have rejected the approach. Issue #8604 is an open bug. However, the contributor should ideally comment on the issue first to announce intent to submit a fix and describe the approach, before opening the PR. This gives maintainers a chance to redirect if needed.

### 4. Huge PR / mixed concerns
**PASS.** The PR changes exactly one function in one file (`src/server.cc`), plus adds one test. This is the minimal possible change.

### 5. No repro / no failing test
**PASS (conditional).** The test suite includes a test that specifically reproduces the #8604 symptom (Test 2: kill stub → assert `/v2/health/ready` is non-200). This test would fail on unpatched `main` and pass with the patch. However, this has not been verified against actual Triton — only designed and reviewed.

### 6. Not running tests / pushing failing CI
**FAIL.** The patch has never been compiled against the actual `triton-inference-server/core` source tree. No existing `L0_*` tests have been run. No live Triton instance has been used to verify. This is a blocking requirement before submission.

### 7. Poor PR/issue description
**PASS.** The PR description includes: the bug (with before/after table), the root cause, the fix (with code), the call chain, performance analysis, backward compatibility notes, related issues, type of change, and test plan. It is clear enough for a maintainer unfamiliar with #8604.

### 8. Duplicate issues / ignoring templates
**NEEDS VERIFICATION.** No existing PR on `triton-inference-server/core` was found addressing #8604 during this review, but this should be re-verified immediately before submission. The `core` repo may have a PR template that must be followed.

### 9. Wrong venue
**PASS.** The bug manifests at the `/v2/health/ready` endpoint which is implemented in `InferenceServer::IsReady()` in `triton-inference-server/core`. The fix correctly targets the `core` repo. The `server` repo is a build/packaging repo that includes `core` as a submodule.

### 10. Security mistakes
**PASS.** The fix does not expose any new internal state. The health endpoint already returns ready/not-ready — this change only makes the "ready" answer more accurate. There is no timing attack vector — the variable-time `ModelIsReady()` call does not reveal sensitive information beyond what the health endpoint already conveys.

### 11. License/CLA blind spots
**NEEDS VERIFICATION.** NVIDIA's CLA must be signed before the PR can be merged. The contributor should verify this is done. The new code uses the same Apache 2.0 license header as the rest of the codebase (verified in test files).

### 12. AI-generated slop
**NEEDS VERIFICATION.** The code was developed with AI assistance. The contributor must:
1. Understand every line of the patch and be able to explain it in PR review comments.
2. Have manually verified the type chain (`ModelIdentifier::name_` → `const std::string&`).
3. Be able to answer questions about the `goto` pattern, the lock ordering, and the inflight counter.
4. Not submit until they have compiled and tested the code themselves.

### 13. Submitting to wrong repository
**NEEDS VERIFICATION.** The patch is currently on a fork of `triton-inference-server/server`. The actual change must go to `triton-inference-server/core`. The contributor must fork `core`, create a branch, apply the patch to `src/server.cc`, and submit the PR there.

**Section 6 Verdict: NOT READY — items 6, 11, 12, 13 are blocking**

---

## Section 7: Blocking Pre-Submission Checklist

1. **Fix the copyright year** in `test.sh` and `test_server_readiness.py`: change `2026` to `2025`.

2. **Remove `sleep 2`** from both Test 2 and Test 4 in `test.sh` — the Python polling loop already handles timing.

3. **Consider replacing `import requests` with `tritonclient.http`** in `test_server_readiness.py` for consistency with the Triton QA test suite. Alternatively, verify `requests` availability in the CI container and document the dependency.

4. **Fork `triton-inference-server/core`** (not `server`). The current fork is of the `server` repo. The patch must be submitted to `core`.

5. **Clone the `core` repo locally** and apply the patch to `src/server.cc` in the actual codebase (not as a `.patch.cpp` file, but as an in-place edit of the function).

6. **Compile the patched `core` repo** against the Triton build system. Verify zero compile errors with both Debug and Release configurations.

7. **Run the existing `L0_*` test suite** (or at least `L0_backend_python`) against the patched build to verify no regressions.

8. **Run the new `test.sh`** against a live Triton instance with the patched `core`. Verify all 4 test cases pass.

9. **Verify no existing PR** on `triton-inference-server/core` addresses #8604.

10. **Comment on issue #8604** describing the intended fix approach before opening the PR. Wait for any maintainer feedback.

11. **Sign NVIDIA's CLA** if not already done.

12. **Remove all references to "Track A", "Track B", internal fork branch names, and the internal validation process** from the PR description. The upstream PR should read as a standalone, first-party contribution.

13. **Place the test files** in the correct location within the `core` repo's test infrastructure (verify where Python backend tests live — likely `qa/` in the `server` repo, not in `core`). The test may need to be submitted as a separate PR to the `server` repo, or the `core` PR should reference how to test it.

14. **Review the PR title and body** against the `core` repo's recent merged PRs for formatting conventions.

---

## Section 8: Recommended PR Title, Body, and Commit Message

### PR Title
```
Fix server readiness to check backend instance health under strict readiness (#8604)
```

### Commit Message
```
Fix server readiness to check backend instance health

Under --strict-readiness=true (the default), InferenceServer::IsReady()
previously only checked ModelStates() lifecycle state. A model whose
backend instances had failed at runtime (e.g., Python backend stub
process died) would remain in READY lifecycle state, causing
/v2/health/ready to return 200 while the model could not serve inference.

Add a ModelIsReady() call for each model in READY lifecycle state during
the strict readiness evaluation. ModelIsReady() calls model->IsReady()
which invokes TRITONBACKEND_ModelInstanceReady for each instance,
detecting runtime failures like dead stub processes.

Performance impact is negligible: TRITONBACKEND_ModelInstanceReady in the
Python backend does waitpid(WNOHANG), a non-blocking microsecond-level
check. Health probes typically run every 5-30 seconds.

Non-Python backends that do not implement TRITONBACKEND_ModelInstanceReady
return success by default and are unaffected.

Fixes: triton-inference-server/server#8604
```

### Lines to Remove/Rewrite in Current PR Description

1. Remove the "Track B" label and any reference to "Track A" — upstream has no context for these.
2. Remove the "Branch: `fix-8604-server-readiness` on fork `itsnothuy/server`" line — this is internal.
3. Remove the "Target upstream repo" line — this is obvious from the PR target.
4. The code block in "The Fix" section uses `ms.first`, `ms.second.second`, `ready_state` — these do not match the actual patch which uses `mv`, `vs`, `model_ready`, `*ready`. **Rewrite the code sample to match the actual patch exactly.**
5. Change "Fixes #8604" to "Fixes triton-inference-server/server#8604" since the issue is on the `server` repo but the PR is on `core`.

---

## Overall Verdict

**SUBMIT AFTER 6 CHANGES:**

1. **Fix copyright year** (`2026` → `2025`) in both test files.
2. **Remove `sleep 2`** from `test.sh` (Tests 2 and 4).
3. **Compile the patch** against the actual `triton-inference-server/core` source tree and verify zero errors.
4. **Run existing tests** (`L0_backend_python` at minimum) to verify no regressions.
5. **Run the new test** against a live patched Triton instance.
6. **Sign NVIDIA CLA** and verify the PR targets `triton-inference-server/core` (not `server`).

**The core patch logic is correct.** The argument types match, the lock ordering is safe, the `goto` is valid C++, the behavioral boundaries are preserved, and the performance impact is negligible. The code is minimal, well-commented, and follows the existing patterns in `server.cc`. The only blocking issues are process/infrastructure items (compilation verification, CLA, correct repo targeting) and minor test hygiene (copyright year, unnecessary sleep).

**No code-level defects were found that would cause a maintainer to reject the patch on technical grounds.**
