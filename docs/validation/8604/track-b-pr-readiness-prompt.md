# Prompt: Final PR-Readiness Validation of Track B for Triton Issue #8604

> **Target model:** Claude Opus 4.6  
> **Purpose:** Last-pass validation of Track B (`fix-8604-server-readiness`) before opening a real upstream PR to `triton-inference-server/core`  
> **Date written:** 2026-02-24  
> **Author context:** Solo contributor, fork `itsnothuy/server`, no prior merge history with upstream Triton maintainers. Code was developed with AI assistance — submitter is responsible for owning every line.

---

## System Prompt

You are a senior C++ systems engineer and open-source maintainer who has reviewed hundreds of PRs for large infrastructure projects (LLVM, Kubernetes, gRPC, Triton Inference Server). You review code with the assumption that the submitter is a capable but unknown external contributor. You do not give benefit of the doubt — you flag every specific defect.

You speak precisely. You do not hedge. When you say something is wrong, you cite the exact line or mechanism. When you say something is correct, you explain why. You are not trying to be encouraging; you are trying to prevent a low-quality PR from being submitted to an upstream project where it would waste maintainer time.

---

## The Bug Being Fixed: Issue #8604

The Triton Inference Server Python backend runs each `model.py` in a child process called the **stub process** (`triton_python_backend_stub`). Communication between the parent Triton server and the stub uses `boost::interprocess` shared-memory queues and a health mutex.

**When the stub process dies** (crash, OOM kill, SIGKILL, segfault in user `model.py`):

- `ModelReadyState` remains `READY` in the lifecycle state machine — it is never updated.
- `IsStubProcessAlive()` (`src/python_be.cc` lines 155–171) will return `false` on next call via a 1-second health mutex timeout or a `waitpid(WNOHANG)` check (`StubActive()`).
- The per-model endpoint `/v2/models/{model}/ready` already returns **503** correctly — it calls `ModelIsReady()` → `model->IsReady()` → `TRITONBACKEND_ModelInstanceReady` → `IsStubProcessAlive()`.
- The **server-level endpoint `/v2/health/ready` returns 200 incorrectly** — it calls only `InferenceServer::IsReady()` which checks `ModelStates()` (the lifecycle enum), never `ModelIsReady()`.

This is the sole reported symptom of #8604: Kubernetes readiness probes hit `/v2/health/ready` and see a healthy server even when all inference requests to the model fail.

---

## The Track B Fix (Complete Implementation)

**Branch:** `fix-8604-server-readiness` on fork `itsnothuy/server`  
**Target upstream repo:** `triton-inference-server/core`  
**File to be modified:** `src/server.cc`  
**Function:** `InferenceServer::IsReady()`

### The Actual Patch (full patched function)

```cpp
Status
InferenceServer::IsReady(bool* ready)
{
  *ready = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    return Status(Status::Code::UNAVAILABLE, "Server exiting");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  // Server is considered ready if it is in the ready state.
  // Additionally can report ready only when all models are ready.
  *ready = (ready_state_ == ServerReadyState::SERVER_READY);
  if (*ready && strict_readiness_) {
    // Strict readiness... get the model status and make sure all
    // models are ready.
    const auto model_versions = model_repository_manager_->ModelStates();

    for (const auto& mv : model_versions) {
      // If a model status is present but no version status,
      // the model is not ready as there is no proper version to be served
      if (mv.second.size() == 0) {
        *ready = false;
        goto strict_done;
      }
      for (const auto& vs : mv.second) {
        // Okay if model is not ready due to unload
        if ((vs.second.first != ModelReadyState::READY) &&
            (vs.second.second != "unloaded")) {
          *ready = false;
          goto strict_done;
        }

        // For models in READY lifecycle state, also check runtime
        // backend instance readiness. This catches cases where the
        // model was loaded successfully (lifecycle = READY) but the
        // backend has become unhealthy at runtime (e.g., Python backend
        // stub process died).
        //
        // Without this check, /v2/health/ready returns 200 even when a
        // model cannot serve inference — the core symptom of issue #8604.
        //
        // Note: ModelIsReady() calls model->IsReady() which invokes
        // TRITONBACKEND_ModelInstanceReady for each instance. In the
        // Python backend, this calls StubActive() -> waitpid(WNOHANG),
        // which is non-blocking and microsecond-level.
        if (vs.second.first == ModelReadyState::READY) {
          bool model_ready = false;
          Status status =
              ModelIsReady(mv.first.name_, vs.first, &model_ready);
          if (!status.IsOk() || !model_ready) {
            LOG_VERBOSE(1) << "Model '" << mv.first.name_ << "' version "
                           << vs.first
                           << " is in READY lifecycle state but failed "
                              "runtime readiness check during server "
                              "readiness evaluation";
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

### The Call Chain Being Added

```
InferenceServer::IsReady()
  → ModelIsReady(name, version, &ready)        [src/server.cc — existing method]
    → model->IsReady()                          [per-instance loop in triton core]
      → TRITONBACKEND_ModelInstanceReady()      [backend C API]
        → ModelInstanceState::StubActive()     [python_backend/src/python_be.cc]
          → waitpid(stub_pid_, WNOHANG)         [Linux syscall, non-blocking]
```

### Key Design Claims Made in PR Description

1. **Performance is negligible**: `waitpid(WNOHANG)` is ~1–2 µs. Health probes run every 5–30s.
2. **Backward compatible**: Only affects `--strict-readiness=true` path. `strict_readiness_=false` is unchanged.
3. **Non-Python backends unaffected**: Backends without `TRITONBACKEND_ModelInstanceReady` return `true` by default.
4. **Fails closed, not open**: If `ModelIsReady()` returns an error, `*ready` is set to `false` — safe failure mode.

### PR Description (verbatim, for your review)

```
PR Title: fix: server readiness endpoint detects dead Python backend stubs (#8604)

Summary: Fixes /v2/health/ready to reflect backend runtime health when
--strict-readiness=true (the default).

The Bug:
  /v2/models/{model}/ready → Returns 503 correctly (unchanged)
  /v2/health/ready         → Was: 200 WRONG. Now: 503 correct.

The Fix: In InferenceServer::IsReady() (src/server.cc), for each model in
READY lifecycle state, additionally call ModelIsReady() to verify the backend
reports the model is actually operational.

Related Issues:
  Fixes #8604
  Related: #7230, #7588

Type of Change: Bug fix (non-breaking)

Test Plan:
  New: fix-8604/core/tests/server_readiness/test.sh — 4 test cases
  Existing: qa/L0_backend_python/model_readiness/test.sh — already passes
```

### Test Suite

**`test.sh` (shell integration test, 4 test cases):**
1. Start Triton + Python model with `--strict-readiness=true` → assert `/v2/health/ready` returns 200
2. Kill stub process with `kill -9` → poll up to 10s → assert `/v2/health/ready` returns non-200
3. Assert `/v2/models/identity_fp32/ready` also returns non-200 (baseline — already worked)
4. Restart server with `--strict-readiness=false`, kill stub → assert `/v2/health/ready` still returns 200

**`test_server_readiness.py` (Python client, uses `requests`):**
- 4 `unittest.TestCase` methods matching the 4 shell test cases above
- Polling loops with 0.5s sleep and 10s deadline (not bare `sleep`)
- Failure messages cite the specific #8604 symptom when assertions fail

**Identity model used for testing (`identity_fp32`):**
- `backend: "python"`, `max_batch_size: 0`, single fp32 input/output passthrough
- Created inline in `test.sh` via heredoc, no external dependency

---

## Your Task

Perform a final, exhaustive PR-readiness validation of Track B. This is a **blocking gate review** — if you find anything that would cause a maintainer to request changes, close the PR, or label it as low-quality, say so now so it can be fixed before submission.

Your review must cover all eight sections below. For each, give an explicit **READY / NOT READY / NEEDS VERIFICATION** verdict with a one-to-three sentence justification. Do not write "looks good" without citing the specific evidence.

---

### Section 1: Code Correctness

Answer each question with a specific yes/no/unknown and the exact evidence:

1. **`ModelIsReady(mv.first.name_, vs.first, &model_ready)` — are the argument types correct?**
   - `mv` is an element of `model_repository_manager_->ModelStates()`. What is the return type of `ModelStates()`? What is `mv.first`? Is `mv.first.name_` the correct way to access the model name, or is it `mv.first` directly, or `mv.first.Name()`?
   - `vs` is an element of `mv.second`. What is `vs.first`? Is it `int64_t` as assumed? Does `ModelIsReady` take `int64_t` for the version or `std::string`?
   - `ModelIsReady` signature in `core/src/server.h` or `core/src/server.cc` — does it match the call?

2. **`goto strict_done` across a local variable initialization** — the patch declares `bool model_ready = false` and `Status status = ModelIsReady(...)` inside a `for` loop body, then uses `goto strict_done` where `strict_done:` is after the loop. Does this `goto` jump over any variable initializations in a way that would trigger a C++ compile error ("jump bypasses variable initialization")?

3. **`if (vs.second.first == ModelReadyState::READY)` logic** — the preceding `if` checks `vs.second.first != ModelReadyState::READY`. If that condition is false (i.e., the state IS `READY`), the code falls through to the new `if (vs.second.first == ModelReadyState::READY)` block. This means: if the lifecycle state is `READY`, always call `ModelIsReady()`. Is this the intended behavior? Is there an "unloaded" edge case where the state is `READY` but calling `ModelIsReady()` would be incorrect or crash?

4. **`ModelIsReady()` — does it hold any locks that `IsReady()` also holds?** If `IsReady()` holds a lock before this call, and `ModelIsReady()` tries to acquire the same lock, there is a deadlock. Verify whether `model_repository_manager_->ModelStates()` acquires a lock that `ModelIsReady()` also needs.

5. **`ScopedAtomicIncrement inflight(inflight_request_counter_)`** — this increments the in-flight request counter for the duration of `IsReady()`. The new code calls `ModelIsReady()` inside this scope. Does `ModelIsReady()` check or react to `inflight_request_counter_` in any way that could cause incorrect behavior?

---

### Section 2: Compilation Readiness

The patch is currently stored as a `.patch.cpp` file on the fork — it has **never been compiled against the actual `triton-inference-server/core` source tree**.

1. Identify every symbol used in the new code block (`ModelIsReady`, `mv.first.name_`, `LOG_VERBOSE`, `ModelReadyState::READY`, `Status`) and state whether each is already included/available in `src/server.cc` at the point of insertion.
2. Is `LOG_VERBOSE` the correct macro for this verbosity level, or should it be `TRITONSERVER_LOG_VERBOSE`? The existing code in this function uses neither — what logging macro does the surrounding code use?
3. The `goto strict_done` pattern with the label `strict_done:;` is used in the *original* code. The new code adds another `goto strict_done` inside a nested `if`. Verify that adding a `goto` inside an `if` block inside a `for` loop is valid C++ and will compile without "jump to label crosses initialization" errors for the `bool model_ready` and `Status status` variables declared in the same block.

---

### Section 3: Behavioral Correctness at the Boundaries

Test each scenario mentally and state the expected behavior with and without the patch:

1. **Scenario: All models healthy.** `ModelIsReady()` returns true for all. Expected: `/v2/health/ready` returns 200. Does the patch preserve this?

2. **Scenario: One model loading (state = `LOADING`).** `ModelStates()` returns `LOADING` for that model. The existing `if ((vs.second.first != ModelReadyState::READY) && ...)` check fires. The new `if (vs.second.first == ModelReadyState::READY)` block is NOT reached. Expected: `/v2/health/ready` returns 503. Does the patch preserve this?

3. **Scenario: One model unloaded (state = `READY`, reason = "unloaded").** The existing check has `vs.second.second != "unloaded"` as a guard — when state is READY and reason is "unloaded", the `if` condition is false and `goto strict_done` is NOT taken. The new block checks `vs.second.first == ModelReadyState::READY` — this is true. So `ModelIsReady()` is called for an unloaded model. **Is this correct?** What does `ModelIsReady()` return for a model that has been unloaded? Does it return false (causing server to report not-ready for an unloaded model, which was previously acceptable) or does it handle unloaded models gracefully?

4. **Scenario: Server has zero models loaded.** `model_versions` is empty. The `for` loop doesn't execute. `*ready` remains `true`. The patch does not affect this. Expected: 200. Is this preserved?

5. **Scenario: `ModelIsReady()` returns a non-OK `Status` (not just `model_ready=false`).** The patch sets `*ready = false` in this case. What are the real-world conditions under which `ModelIsReady()` returns a non-OK Status? Is treating a non-OK Status as "server not ready" always correct, or could it mask a transient error (e.g., the model repository is locked for a brief period)?

---

### Section 4: Performance Analysis

The PR description claims "negligible overhead." Validate this claim:

1. **`ModelIsReady()` call frequency:** `/v2/health/ready` is called by Kubernetes readiness probes. Default probe interval is 10s; minimum is 1s. With N models loaded, `ModelIsReady()` is called N times per probe. Each call does `waitpid(WNOHANG)` per instance. Is there a realistic scenario where this becomes non-negligible (e.g., 1000 models, aggressive health probing at 1s interval)?

2. **`ModelIsReady()` acquires any locks?** If it takes a read lock on the model registry or the model instance map, and this lock is also needed by inference threads, the health probe calls could add contention. Assess whether `ModelIsReady()` → `model->IsReady()` → `TRITONBACKEND_ModelInstanceReady` is lock-free from the perspective of inference threads.

3. **`inflight_request_counter_` increment:** `IsReady()` wraps its body in `ScopedAtomicIncrement`. The new code extends the time this counter is incremented (because `ModelIsReady()` calls take time). Does anything in Triton check `inflight_request_counter_` during shutdown or model unloading in a way that could be affected by this?

---

### Section 5: Test Quality

For each of the 4 test cases in `test.sh`, answer:

1. **Test 1 (initial readiness):** Does `run_server` wait for the server to be fully ready before the test runs? If the test runs before the server finishes loading the model, it will fail spuriously. Is there a health-check polling loop, or does `run_server` from `../common/util.sh` already handle this?

2. **Test 2 (stub killed, server non-ready):** The test does `kill -9 $STUB_PID && sleep 2`, then runs the Python test. The Python test polls for up to 10 seconds. **Is `sleep 2` necessary, or is it masking a race condition?** More precisely: after `kill -9`, the stub process is dead. The next call to `IsStubProcessAlive()` returns false. But `IsReady()` is only called when a health probe arrives — it doesn't proactively poll. The Python test polls `/v2/health/ready` every 0.5s. Does `sleep 2` serve any purpose that the Python test's polling loop doesn't already cover? Removing it would make the test faster. Leaving it creates a false sense that 2s is "required."

3. **Test 3 (per-model readiness):** This test runs *after* the stub is already killed from Test 2, with no restart in between. Is there a risk that the server has already entered an error state that causes Test 3 to fail for a different reason than expected (e.g., the server crashed after stub death)?

4. **Test 4 (strict=false):** The test restarts the server with `--strict-readiness=false`. Then kills the stub. Then asserts the server is still ready. **Critical gap:** between restarting the server and killing the stub, is there a wait for the server to finish loading the model? If the stub is killed before the model finishes loading, the test may pass for the wrong reason.

5. **`pgrep -f "triton_python_backend_stub.*identity_fp32"` — is this reliable?** If the process name or arguments change between Triton versions, this grep will fail silently (returns empty `$STUB_PID`). The test handles the empty case with an error message, but is there a more robust way to get the stub PID (e.g., from Triton's own API or from a known parent PID)?

6. **The Python test client uses `import requests` — is this available in the Triton QA container?** The existing tests under `qa/L0_backend_python/` use `tritonclient.http` and `tritonclient.grpc`. Does the `requests` library follow the same assumption, or should the test use `tritonclient.http` for consistency with the rest of the test suite?

---

### Section 6: New Contributor Checklist Audit

For each item in the checklist below, give **PASS**, **FAIL**, or **NEEDS VERIFICATION** with one sentence of justification specific to Track B:

1. **Skipping CONTRIBUTING.md / Code of Conduct** — Does the PR follow all required format, tests, labels, and communication norms for `triton-inference-server/core`? Has the submitter read the repo's contributing guidelines and Code of Conduct?

2. **Drive-by "random changes"** — Is this a genuine, necessary bug fix, or is it a cosmetic/low-value change? Is the motivation clearly documented?

3. **Opening a PR without checking if the project wants the change** — Is this fix consistent with the Triton team's stated direction? Is there any evidence that maintainers have already decided to fix this differently? Does the `core` repo have its own contributing guidelines that might be more specific than the `server` repo?

4. **Huge PR (too many files / mixed concerns)** — Does the PR change only what is necessary? Are there any changes that should be in a separate PR?

5. **No repro / no failing test** — Does the test suite include a test that fails on the unpatched `main` and passes with the patch applied? Is the repro step documented in the PR description?

6. **Not running tests / pushing failing CI** — Has the patch been compiled against the actual `triton-inference-server/core` source? Have the existing `L0_*` tests been run? Has the new test been run on a live Triton instance?

7. **Poor PR/issue description** — Does the PR description contain: the bug, the root cause, the fix, the before/after behavior, and how to reproduce and test? Is it clear enough for a maintainer who has never seen issue #8604?

8. **Duplicate issues / ignoring templates** — Is there an existing PR on `triton-inference-server/core` that already addresses this? Does the `core` repo have a PR template that must be followed?

9. **Wrong venue** — Should this be filed against `triton-inference-server/core` or against `triton-inference-server/server`? The bug manifests in the server repo but the `InferenceServer::IsReady()` function lives in the `core` repo. Is this the correct target?

10. **Security mistakes** — Does the fix introduce any security surface (e.g., exposing internal state via the health endpoint, timing attacks via the variable-time `ModelIsReady()` call)?

11. **License/CLA blind spots** — Does the submitter need to sign NVIDIA's CLA before this PR can be merged? Has this been done? Does the `core` repo use the Apache 2.0 license, and does the new code comply?

12. **AI-generated slop** — This code was developed with AI assistance. What specific evidence should the submitter provide to demonstrate they understand and own every line? What verification steps are required before this can be submitted responsibly?

13. **Submitting to the wrong repository** — The patch is currently on a fork of `triton-inference-server/server`. The actual change goes to `triton-inference-server/core`. Has the submitter verified they are opening the PR against the correct upstream repo and branch (`main` or a specific release branch)?

---

### Section 7: What Is Missing Before This Can Be Submitted

Produce an ordered checklist of every action that must be completed before the PR can be opened. Format each item as an action verb ("Clone...", "Apply...", "Run...", "Verify...", "Sign..."). Do not list items that are already done. Do not list nice-to-haves — only blocking requirements.

---

### Section 8: Recommended PR Title, Body, and Commit Message

Based on your review, provide:

1. **Final recommended PR title** — following the `triton-inference-server/core` convention (look at recent merged PRs for naming patterns — imperative present tense, lowercase, issue number in parentheses).

2. **Final recommended commit message** — one subject line (≤72 chars) + blank line + body (what changed, why, and the issue reference). Use the upstream commit style.

3. **Any line in the current PR description that should be removed or reworded** — be specific. If the description references "Track B" or the internal fork branch name, those should be removed before submission to a public upstream PR.

---

## Output Format

```
## Section 1: Code Correctness
[question-by-question answers with READY/NOT READY/NEEDS VERIFICATION]

## Section 2: Compilation Readiness
[analysis]

## Section 3: Behavioral Correctness at the Boundaries
[scenario-by-scenario analysis]

## Section 4: Performance Analysis
[analysis]

## Section 5: Test Quality
[question-by-question answers]

## Section 6: New Contributor Checklist Audit
[PASS/FAIL/NEEDS VERIFICATION per item]

## Section 7: Blocking Pre-Submission Checklist
[ordered action list]

## Section 8: Recommended PR Title, Body, Commit Message
[final text]

## Overall Verdict
[single paragraph: SUBMIT AS-IS / SUBMIT AFTER N CHANGES / DO NOT SUBMIT — with the N changes listed explicitly]
```

Do not pad. Do not soften. If the patch has a real compilation error, say "this will not compile because X." If a test has a race condition, say "this test has a race condition because Y." The goal is to make this PR as strong as possible before it reaches an upstream maintainer, not to validate effort already spent.
