# Track A vs Track B — Executive Summary

> **Source:** Claude Opus 4.6 analysis, 2026-02-24  
> **Context:** Side-by-side comparison of two proposed fixes for `triton-inference-server` Issue #8604

---

Track B is the correct fix for Issue #8604. Track A is a valuable enhancement but is not the fix — it is new functionality that masks the root cause while introducing significant complexity. Track B surgically corrects the one broken code path (`InferenceServer::IsReady()` not calling `ModelIsReady()`) with ~15 lines, zero inference-path impact, and zero new failure modes. Track A adds ~120 lines of production code with a new mutex, atomic state, rate limiting, and process lifecycle management in the inference hot path — a contribution that CONTRIBUTING.md explicitly says requires prior design discussion with maintainers. Submit Track B as a PR now. Post Track A as a design proposal on Issue #8604 and wait for maintainer feedback before writing the PR.

---

## Dimension 1: Correctness and Logical Necessity

**Does Track A fix the root cause of #8604?**
No. Issue #8604 is "`/v2/health/ready` returns 200 when a Python backend stub is dead." Track A does not touch `/v2/health/ready` or `InferenceServer::IsReady()`. After Track A is applied, if the stub dies and restarts within 2–3 seconds, `/v2/health/ready` still returns 200 during that entire dead window — the exact symptom reported. Track A mitigates the *consequence* (broken model stays broken) but does not fix the *reported bug* (health endpoint lies).

**Does Track B fix the root cause of #8604?**
Yes. The root cause is that `InferenceServer::IsReady()` checks `ModelStates()` (lifecycle enum, never updated on stub death) instead of `ModelIsReady()` (which calls `TRITONBACKEND_ModelInstanceReady` → `IsStubProcessAlive()`). Track B adds exactly the missing `ModelIsReady()` call. After the patch, `/v2/health/ready` returns 503 when a stub is dead. This is the precise fix for the reported symptom.

**Which is more logically necessary?**
Track B. If a maintainer reads Issue #8604 — "health endpoint reports ready when model cannot serve" — the fix is making the health endpoint check runtime health. That is Track B. Track A is an enhancement: "auto-restart dead stubs." That's a separate feature request, not the fix for #8604.

**Is Track B ever wrong?**
One potential false-positive: if `ModelIsReady()` takes longer than expected (e.g., a backend with a slow `TRITONBACKEND_ModelInstanceReady` implementation), `/v2/health/ready` latency increases. But for Python backend, `StubActive()` does `waitpid(WNOHANG)` — non-blocking, 1–2 µs. For backends that don't implement `TRITONBACKEND_ModelInstanceReady`, the default returns `true`, so no behavioral change. There is no case where Track B returns 503 when it should return 200: the `ModelIsReady()` call only runs for models in `READY` lifecycle state, and it simply invokes the same logic that `/v2/models/{model}/ready` already uses correctly.

**Is Track A ever wrong?**
Two edge cases:

1. **Restart during shutdown:** If the server is shutting down and a request hits a dead stub, Track A will attempt to restart the stub while the destructor is tearing down `ModelInstanceState`. The `try_to_lock` pattern helps, but `LaunchStubProcess()` during shutdown is undefined behavior — the model repository manager may have already cleaned up resources.
2. **Restart when `model.py` is fundamentally broken:** If `model.py` crashes in `initialize()`, Track A will restart the stub, which will crash again in `initialize()`, consuming 5 restarts in rapid succession before rate-limiting. During those 5 restarts, each `LaunchStubProcess()` does shared memory allocation, process fork, Python interpreter startup — substantial work for a guaranteed-to-fail outcome.

**Does `/v2/models/{model}/ready` already working make Track B redundant?**
No — complementary. The per-model endpoint already calls `ModelIsReady()`. The server-level endpoint does not. They serve different consumers: Kubernetes readiness probes typically hit `/v2/health/ready` (server-level), not per-model endpoints. Track B closes the gap by making the server-level endpoint as accurate as the per-model one.

---

## Dimension 2: Complexity, Maintenance Burden, and Risk

### Track A

**Production LOC delta:** ~120 lines (2 new methods, modified `TRITONBACKEND_ModelInstanceExecute`, 6 new header members).

**New failure modes introduced:**

1. `RestartStubProcess()` calls `TerminateStub()` → `Stub()->ClearQueues()` → `Stub().reset()` → `LaunchStubProcess()`. If `LaunchStubProcess()` fails (port conflict, shared memory exhaustion, Python not found), the instance is in a state where `Stub()` has been reset but not re-created. Any subsequent request will segfault on `Stub()->` dereference.
2. `restart_mutex_` with `try_to_lock` means if one thread is mid-restart, other threads silently fail the lock and fall through to fail their batch. If the restart takes longer than the next health check interval, the load balancer may observe a temporarily-inconsistent state.
3. `restart_in_progress_.exchange(true)` is checked *after* `try_to_lock` succeeds, but `restart_in_progress_` is set to `false` *outside* the lock scope. There is a window where `restart_in_progress_` is `false` but the restart has not fully committed (the lock is released but the new stub is not yet fully initialized). A concurrent thread could attempt another restart in this window.
4. `restart_count_` is `std::atomic<int>` but `restart_window_start_` is not atomic. `CanRestart()` reads both — under the mutex this is safe, but if `CanRestart()` is ever called outside the mutex, there's a data race on `restart_window_start_`.
5. If the monitor thread detects stub death concurrently with a request thread, both may attempt cleanup. The monitor thread calls `TerminateMonitor()` and related cleanup — this races with `RestartStubProcess()`.

**Concurrency analysis:**
The `try_to_lock` pattern is a reasonable first approach, but the interaction between `restart_mutex_`, `restart_in_progress_` (atomic), and the existing `health_mutex_` (boost::interprocess) creates a three-lock system with no documented ordering. In a server with multiple model instances and concurrent requests, the potential for subtle ordering issues over 2–3 years of development is significant.

**State machine after restart:**
If `LaunchStubProcess()` succeeds, `thread_pool_`, `request_executor_`, and `Stub()` should all be consistent — `LaunchStubProcess()` creates all of them. But the old `thread_pool_` may have pending tasks that reference the old `Stub()` (now destroyed). The code calls `Stub().reset()` before `LaunchStubProcess()`, but does not wait for pending thread pool tasks to complete. The comment in the patch says "Do NOT call `thread_pool_->wait()` here" because tasks may be stuck on dead-stub IPC — this is correct reasoning, but it means abandoned tasks may still hold references to destroyed objects.

**Maintenance burden: 4/5** (high). Touches inference hot path, introduces new concurrency primitives, interacts with process lifecycle management that NVIDIA actively develops.

### Track B

**Production LOC delta:** ~15 lines (one `if` block inside an existing `for` loop).

**New failure modes introduced:**

1. If a backend's `TRITONBACKEND_ModelInstanceReady` implementation is slow or blocks, `/v2/health/ready` response time increases. Mitigated: Python backend's implementation is `waitpid(WNOHANG)` — non-blocking. Default implementation returns `true` immediately.
2. If `ModelIsReady()` itself throws or crashes, `IsReady()` could fail. Mitigated: `ModelIsReady()` returns a `Status` object, and the patch checks `!status.IsOk()` — any error safely results in `ready = false`.

That's it. No new locks, no new state, no new process management.

**Maintenance burden: 1/5** (minimal). The added code is a straightforward call to an existing API (`ModelIsReady`) inside an existing loop. Any future changes to `ModelIsReady` or `TRITONBACKEND_ModelInstanceReady` will automatically flow through. The `goto strict_done` pattern matches the existing code style exactly.

---

## Dimension 3: Long-Term Value

**Kubernetes production value:**
Track B is immediately valuable to every Kubernetes operator running Triton with `--strict-readiness=true` (the default). It makes readiness probes accurate, which is the foundational requirement for safe traffic routing. Without Track B, operators cannot trust `/v2/health/ready` — which means they either (a) don't use readiness probes (dangerous) or (b) write custom per-model health checks (expensive).

Track A is valuable *after* Track B, as a resilience enhancement. In a Kubernetes environment with HPA and pod disruption budgets, auto-restart is nice-to-have — the standard pattern is: readiness fails → K8s stops routing → liveness fails → K8s restarts the pod. Self-healing at the stub level is a faster recovery path, but it requires accurate readiness signaling (Track B) to be safe.

**Self-healing masking problems:**
Track A can mask real issues. If `model.py` leaks memory and crashes every 30 minutes, Track A will restart it silently up to 5 times per minute. An operator checking logs might see "Successfully restarted stub process" but not connect it to the underlying leak. Without Track A, the model goes permanently unhealthy → the operator investigates → the root cause is found. There is a legitimate architectural debate about whether inference servers should self-heal or fail-loud.

**Future NVIDIA implementation:**
The `TODO: Implement restart on decoupled` comment in upstream code (PR #360) strongly suggests NVIDIA intends to implement their own restart mechanism. Track A would create a merge conflict and a maintenance burden if NVIDIA's design differs (which it almost certainly will — they may want integration with model loading/unloading lifecycle, metrics, and the monitor thread). Track B is unlikely to conflict because it uses an existing public API (`ModelIsReady`) in its intended way.

**Acceptance likelihood:**

- **Track B:** High. It calls an existing function that is already used by the per-model readiness endpoint. The fix is obviously correct. A reviewer can verify it in 5 minutes.
- **Track A:** Low as-is. It introduces new concurrency, process lifecycle management, and rate limiting. Any maintainer will want to discuss the design first. It will likely get "we like the intent but this needs redesign" feedback — not because the code is bad, but because the design decisions (rate limits, lock strategy, interaction with monitor thread) should be made by the team that maintains the code long-term.

---

## Dimension 4: Cost

**Implementation effort:**

- Track B: ~2 hours. One function, one code path, existing API.
- Track A: ~20 hours. Two new methods, header changes, modified hot path, new concurrency primitives, rate limiting logic, extensive testing of edge cases.
- Ratio: **10:1**.

**Review cost to maintainers:**

- Track B: 1 review round. A maintainer reads the 15-line diff, verifies `ModelIsReady` does what it claims, checks the test, approves. 30 minutes.
- Track A: 3–5 review rounds minimum. Concurrency review alone will take multiple iterations. Questions about locking order, interaction with shutdown, interaction with monitor thread, rate limit parameters, state machine correctness post-restart. Each round takes 1–2 weeks of elapsed time for an external contributor. Expect 2–3 months to merge.

**Blast radius of a Track A bug:**
If `RestartStubProcess()` has a race condition: segfault in the inference hot path, corrupted shared memory, orphaned child processes. This affects every Python model on the server. Worst case: data corruption if a half-restarted instance processes a request with stale shared memory references.

**Blast radius of a Track B bug:**
If `ModelIsReady()` false-positives (returns non-ready when the model is actually ready): `/v2/health/ready` returns 503 → K8s stops routing traffic → the pod receives no requests. This is a *safe failure mode* — it fails closed, not open. No data corruption, no segfaults, no process leaks. The operator sees the pod is "not ready" and investigates. The worst outcome is unnecessary downtime, not silent corruption.

---

## Dimension 5: Impact on Existing Codebase

**Track A — inference hot path impact:**
Every call to `TRITONBACKEND_ModelInstanceExecute` (every inference request) now executes `restart_in_progress_.load()` — a single atomic load, ~1 ns, effectively zero overhead. The `restart_mutex_` is only *attempted* (`try_to_lock`) in the error path, so the fast path (stub alive) has no mutex contention. Verdict: negligible performance impact on the happy path.

However, the mere *presence* of restart logic in the execute path increases cognitive load for anyone reading or modifying `TRITONBACKEND_ModelInstanceExecute` in the future. This function is already complex (decoupled vs. non-decoupled paths, request batching, IPC). Adding restart logic makes it harder to reason about.

**Track A — locking order:**
`restart_mutex_` is new. The existing locks in python_backend are: `health_mutex_` (boost::interprocess, per-instance), `Stub()->StubMessageQueue()` (IPC queue lock). `RestartStubProcess()` acquires `restart_mutex_` and then calls `TerminateStub()` which may interact with `health_mutex_`. If `health_mutex_` is held by another thread doing `IsStubProcessAlive()`, there's no deadlock because `IsStubProcessAlive()` uses a timed lock (1s timeout). But the interaction is subtle and undocumented.

**Track B — inference latency:**
Zero. `IsReady()` is only called by the HTTP/gRPC handler for `/v2/health/ready`. It is never called in the inference request path. Inference requests go through `ModelIsReady()` for per-model checks, but `IsReady()` (server-level) is not in that path.

**Track B — non-Python backends:**
`TRITONBACKEND_ModelInstanceReady` is an optional backend API. Backends that don't implement it have the default stub that returns `nullptr` (meaning "ready"). Track B calls `ModelIsReady()` which calls `model->IsReady()` which iterates instances and calls `TRITONBACKEND_ModelInstanceReady`. For TensorRT, ONNX, OpenVINO: they don't implement it → default returns true → no behavioral change. Only Python backend (and any future backend that implements the API) is affected.

**`--strict-readiness=false`:**

- Track A: still active. Restart happens regardless of readiness mode (it's triggered by inference failure, not readiness checks).
- Track B: completely inert. The new code is inside the `if (*ready && strict_readiness_)` block.

---

## Dimension 6: PR Submission Readiness

### Track A

| # | Check | Verdict | Justification |
|---|-------|---------|---------------|
| 1 | Linked issue | **PASS** | PR description references "Fixes #8604" and cites #7230, #7588 |
| 2 | Scoped to one concern | **PASS** | Only adds restart logic, all changes serve that goal |
| 3 | No commented-out code | **PARTIAL** | The patch file contains extensive inline comments explaining design rationale — acceptable, but the `// Do NOT call thread_pool_->wait() here` and `// Do NOT call UpdateHealth()` comments read as defensive justification, not documentation. These should be shortened to state the invariant, not argue for it |
| 4 | Build log clean | **FAIL** | The patch is delivered as `.patch.cpp` files with inline comments about insertion points. It has not been applied to the actual source tree and compiled. Method signatures (`TerminateStub()`, `ClearQueues()`) are assumed but not verified against current upstream. `Stub().reset()` — is `Stub()` a `unique_ptr` or `shared_ptr`? If `shared_ptr`, `.reset()` may not destroy the object immediately. Unverified |
| 5 | Tests behavioral | **PASS** | Tests check HTTP status codes and model behavior, not log strings. Polling loops instead of `sleep`. Rate limit test verifies permanent failure |
| 6 | PR description quality | **PASS** | `PR_DESCRIPTION.md` contains motivation, before/after table, comparison with PR #431, file list, test plan |
| 7 | CONTRIBUTING.md compliance | **FAIL** | CONTRIBUTING.md: "Contributions intended to add significant new functionality must follow a more collaborative path. Before submitting a large PR... submit a GitHub issue that describes the proposed change." No design issue has been opened. Auto-restart is significant new functionality |
| 8 | Not a duplicate | **PARTIAL** | PR #431 addresses the same problem space. The PR description should not criticize #431 directly — it should state "supersedes #431" and explain the different approach neutrally. The current description contains a comparison table that could read as adversarial |
| 9 | Not AI slop | **FAIL** | The code was generated with AI assistance (documented in conversation history). The patch has not been compiled, has not been applied to the actual source tree, and has not been tested against a running Triton instance. Methods like `TerminateStub()`, `ClearQueues()`, `TerminateMonitor()` are called but their exact signatures and behavior have not been verified against current upstream. This is the definition of "plausible-looking PRs that are wrong, untested" from Trap 13 |
| 10 | Design discussed first | **FAIL** | No GitHub issue or discussion has been opened. This is explicitly required by CONTRIBUTING.md for "significant new functionality" |

### Track B

| # | Check | Verdict | Justification |
|---|-------|---------|---------------|
| 1 | Linked issue | **PASS** | References "Fixes #8604" |
| 2 | Scoped to one concern | **PASS** | One function, one bug, one fix |
| 3 | No commented-out code | **PASS** | The patch file shows both original and patched code for context, but the actual patch is clean. The `LOG_VERBOSE(1)` is operational logging, not debug |
| 4 | Build log clean | **PARTIAL** | The patch calls `ModelIsReady(mv.first.name_, vs.first, &model_ready)`. Need to verify: (a) `mv.first.name_` is the correct field for the model identifier — `mv.first` is a `ModelIdentifier`, and `.name_` may need to be verified as the correct accessor. (b) `vs.first` is the version (int64_t). (c) `ModelIsReady` takes `(const std::string&, int64_t, bool*)`. These need compilation verification, but the types are straightforward |
| 5 | Tests behavioral | **PASS** | Tests check HTTP status codes, poll for state changes, test both strict=true and strict=false modes |
| 6 | PR description quality | **PASS** | Clear motivation, before/after table, call chain, performance analysis, file list |
| 7 | CONTRIBUTING.md compliance | **PARTIAL** | This is a bugfix, not "significant new functionality," so prior design discussion is not required. However: CLA must be signed, pre-commit hooks must pass, and the patch must be applied to the actual `triton-inference-server/core` repo and compiled — none of which has been done |
| 8 | Not a duplicate | **PASS** | PR #431 is on `python_backend`, not `core`. Track B targets a different repo entirely. No overlap |
| 9 | Not AI slop | **PARTIAL** | The code was AI-generated but the logic is trivially verifiable: it calls `ModelIsReady()` (an existing function) with the same arguments used by the per-model readiness endpoint (which already works). A human can verify correctness in 5 minutes. However, the patch has not been compiled against the actual `core` source tree. The submitter must: (1) clone `triton-inference-server/core`, (2) apply the patch, (3) compile, (4) run the existing test suite, (5) run the new test — before submitting |
| 10 | Design discussed first | **PASS** | Bugfix, not new functionality. Issue #8604 exists and clearly describes the problem. No prior design discussion needed |

---

## Dimension 7: Final Verdict and Recommended Action

**1. Which track should be submitted first?**
Track B. It is the direct fix for the reported bug. It is small, safe, and reviewable. Track A is an enhancement that should go through design discussion first.

**2. Should Track A be submitted as a PR?**
No, not yet. Track A should be posted as a **design proposal comment on Issue #8604**. Outline the approach (restart with rate limiting, concurrency protection, no retry), link to the branch for reference, and ask for maintainer feedback on whether auto-restart is the desired architecture. The TODO comment in upstream code suggests NVIDIA has opinions about this — get those opinions before writing the final PR.

**3. Is Track B PR-ready as written?**
Almost. Required changes before submission:

1. Apply the patch to actual `triton-inference-server/core` source tree, compile, and run existing tests. The `.patch.cpp` file format is not submittable — it must be a proper diff against the `core` repo.
2. Verify `mv.first.name_` is the correct accessor for `ModelIdentifier`. If the field is `name_` (public member) or `Name()` (accessor method), use the correct one.
3. Sign the CLA and send to `triton-cla@nvidia.com`.
4. Run pre-commit hooks on the modified file.
5. Remove the "ORIGINAL CODE" block from the patch file — the PR should contain only the patched version.

**4. Is there a simpler version of Track A?**
Yes. A minimal stepping stone: instead of auto-restart, just **log a clear error message when stub death is detected** and **set a metric/flag that external monitoring can consume**. This addresses the "silent failure" aspect without introducing restart complexity. The TODO comment already marks where this code should go. This minimal version would be 10–15 lines, require no new mutex, and be uncontroversial to merge.

**5. Overall recommendation:**
Submit Track B as a PR to `triton-inference-server/core` after the 5 pre-submission steps above. Post Track A's design on Issue #8604 as a comment, not a PR. Specifically: describe the restart approach, link to the branch, ask "Is auto-restart the desired architecture for stub process recovery, or does NVIDIA prefer a different approach (e.g., marking the instance permanently failed and relying on external orchestration)?"

Track B fixes the reported bug with minimal risk. It makes the health API honest. Every Kubernetes operator benefits immediately. Track A is valuable but premature — it adds significant complexity to a codebase you don't maintain, touches the inference hot path, introduces concurrency that will need long-term ownership, and directly conflicts with the CONTRIBUTING.md requirement for prior design discussion on new functionality. Submitting Track A as a PR before discussing the design with maintainers is the fastest way to get it closed with a "please open an issue first" comment, burning goodwill.

---

## New Contributor Trap Audit

### Track A

| Trap | Verdict | Detail |
|------|---------|--------|
| 1 — Skipped CONTRIBUTING.md | **FAIL** | CONTRIBUTING.md requires prior design discussion for significant new functionality. Not done |
| 2 — Drive-by cosmetic | **PASS** | This is a real functional change, not cosmetic |
| 3 — Unconfirmed maintainer intent | **FAIL** | No evidence that NVIDIA wants community-contributed auto-restart. The TODO comment suggests they have their own plans |
| 4 — Huge mixed-concern PR | **PASS** | Focused on one feature (restart) |
| 5 — No repro / no failing test | **PASS** | Tests include behavioral verification of restart |
| 6 — Not running CI | **FAIL** | Patch has never been compiled or tested against actual upstream source |
| 7 — Poor PR description | **PASS** | Description is thorough |
| 8 — Duplicate | **PARTIAL** | PR #431 exists in the same space. Must be handled diplomatically — reference it as prior work, not as competition |
| 9 — AI-generated slop | **FAIL** | Code not compiled, method signatures unverified, behavior untested on real Triton instance |
| 10 — Defensive response | **PARTIAL** | PR description's comparison table with PR #431 could read as adversarial. Rewrite to focus on "our approach differs in X, Y, Z" without labeling #431's approach as "critical flaws" |

### Track B

| Trap | Verdict | Detail |
|------|---------|--------|
| 1 — Skipped CONTRIBUTING.md | **PASS** | Bugfix, no prior discussion required |
| 2 — Drive-by cosmetic | **PASS** | Direct fix for a reported bug |
| 3 — Unconfirmed maintainer intent | **PASS** | Issue #8604 exists, bug is clear, fix uses existing API as intended |
| 4 — Huge mixed-concern PR | **PASS** | 15 lines, one concern |
| 5 — No repro / no failing test | **PASS** | Test verifies `/v2/health/ready` returns non-200 after stub death |
| 6 — Not running CI | **FAIL** | Must compile against actual `core` source before submitting |
| 7 — Poor PR description | **PASS** | Clear and complete |
| 8 — Duplicate | **PASS** | No existing PR on `core` for this fix |
| 9 — AI-generated slop | **PARTIAL** | Logic is simple and verifiable, but must be compiled and tested on real infrastructure before claiming it works |
| 10 — Defensive response | **PASS** | No antagonistic content in PR description |

---

## Bottom Line

Do these things in this order: (1) Clone `triton-inference-server/core`, apply the Track B patch as a real source edit to `src/server.cc`, compile, run existing tests, run the new integration test on a live Triton instance with a Python model. (2) Sign and send the CLA to `triton-cla@nvidia.com`. (3) Submit Track B as a PR to `triton-inference-server/core` with the title "fix: /v2/health/ready checks runtime backend health under strict readiness" and "Fixes triton-inference-server/server#8604". (4) Post a comment on Issue #8604 describing the Track A auto-restart design, linking to your fork branch, and asking maintainers whether auto-restart is the desired architecture — do NOT open a PR for Track A until you have maintainer buy-in. (5) Remove all references to PR #431's "critical flaws" from any public-facing text — instead write "this takes a different approach than #431, focusing on X" and let maintainers draw their own conclusions.
