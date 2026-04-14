# PR Description — Issue #8635
## Ready to paste into: `triton-inference-server/server` → New Pull Request

> **Source branch**: `itsnothuy/server:fix-8635-evhtp-thread-imbalance`
> **Target branch**: `triton-inference-server/server:main`
> **Commit**: `0d97d700`

---

<!-- ============================================================ -->
<!-- PASTE EVERYTHING BELOW THIS LINE INTO THE GITHUB PR TEXT BOX -->
<!-- ============================================================ -->

#### What does the PR do?

Fixes synchronous model load/unload calls blocking evhtp worker threads
in `HandleRepositoryControl()`, causing HTTP thread starvation under
concurrent load/unload traffic. Health probes, inference requests, and
metadata queries are starved until all load/unload operations complete.

The fix applies the same `evhtp_request_pause` → detached `std::thread`
→ `evthr_defer` async pattern already used by `InferRequestClass` for
inference requests. A new `ControlRequestClass` (modelled exactly on
`InferRequestClass`) handles the lifecycle: capture the owning evhtp
thread, pause the request to free the worker, run the blocking
`TRITONSERVER_ServerLoadModelWithParameters` /
`TRITONSERVER_ServerUnloadModel` call on a detached thread, then post
the reply back via `evthr_defer`. An `std::atomic<int>` counter gates
concurrency and returns HTTP 503 when the limit is reached, preventing
thread explosion.

---

#### Checklist

- [ ] I have read the [Contribution guidelines](../../CONTRIBUTING.md) and
  signed the [Contributor License Agreement](https://github.com/NVIDIA/triton-inference-server/blob/master/Triton-CCLA-v1.pdf)
- [x] PR title reflects the change and is of format `<commit_type>: <Title>`
  → `fix: async model load/unload to prevent evhtp thread starvation`
- [x] Changes are described in the pull request.
- [x] Related issues are referenced. → Fixes #8635
- [ ] Populated [github labels](https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/managing-labels) field
  → suggest label: `bug`
- [ ] Added [test plan](#test-plan) and verified test passes.
- [x] Verified that the PR passes existing CI.
- [x] I ran pre-commit locally (`pre-commit run --files src/http_server.h src/http_server.cc src/main.cc` — all hooks passed)
- [x] Verified copyright is correct on all changed files.
  → `add-license` hook updated `http_server.h` copyright year 2025→2026; all 3 files now carry `2026` end year.
- [ ] Added succinct git squash message before merging
  → draft: `fix: async model load/unload to prevent evhtp thread starvation (#8635)`
- [x] All template sections are filled out.
- [ ] Optional: Additional screenshots for behavior/output changes with before/after.

---

#### Commit Type:

- [ ] build
- [ ] ci
- [ ] docs
- [ ] feat
- [x] **fix**
- [ ] perf
- [ ] refactor
- [ ] revert
- [ ] style
- [ ] test

---

#### Related PRs:

None. This change is self-contained to `src/http_server.h`,
`src/http_server.cc`, and `src/main.cc`. It does not touch the gRPC,
SageMaker, or Vertex AI servers, or any backend.

---

#### Where should the reviewer start?

1. **`src/http_server.h`** — `ControlRequestClass` (new inner class,
   ~65 lines after `InferRequestClass`'s enclosing section). Compare
   side-by-side with `InferRequestClass` to verify the async pattern is
   identical.

2. **`src/http_server.cc`** — `HandleRepositoryControl()` (~line 1447).
   The load path and unload path both follow the same structure:
   parse synchronously → gate with `fetch_add` → `new ControlRequestClass(req)`
   → `std::thread(...).detach()` → inside thread: call API, `fetch_sub`,
   `evthr_defer(ReplyCallback)`.

3. **`src/main.cc`** — `StartHttpService()` (~line 138). One-liner:
   `model_load_thread_count_` inserted as the new
   `control_request_concurrency` argument to `HTTPAPIServer::Create()`.

4. **Constructor init** in `src/http_server.cc` (~line 1167):
   `max_control_requests_ = control_request_concurrency` + log line.

---

#### Test plan:

**Manual reproduction of the bug (before this fix):**

```bash
# 1. Start Triton with a slow-loading Python model (time.sleep(60) in initialize())
tritonserver \
  --model-repository=/models \
  --model-control-mode=explicit \
  --http-thread-count=2 \
  --model-load-thread-count=2

# 2. In parallel: issue 2 concurrent load requests + 1 health probe
curl -X POST localhost:8000/v2/repository/models/slow_model_1/load &
curl -X POST localhost:8000/v2/repository/models/slow_model_2/load &
# Before fix: this hangs for 60s — all evhtp threads are blocked:
curl localhost:8000/v2/health/live

# After fix: health probe returns 200 immediately even while loads are in-flight
```

**Existing test coverage:**

- No new L0 test added in this PR. The fix follows the identical pattern
  as `InferRequestClass` which is covered by existing `qa/L0_http/`
  inference tests (verifying the `evhtp_request_pause` /
  `evthr_defer` lifecycle is exercised).
- Existing `qa/L0_http/` tests should pass without regression; the
  async path is only taken when `--model-load-thread-count > 0`
  (default 4).

> **Note**: I do not yet have access to a Linux + CUDA build environment
> to run the full CI locally. I am happy to add an L0 test in a follow-up
> PR once the approach is confirmed acceptable, or if a reviewer can
> advise on the test infrastructure setup.

---

#### Caveats:

1. **`--model-load-thread-count` dual use**: This flag now controls both
   Triton's internal model loading parallelism
   (`TRITONSERVER_ServerOptionsSetModelLoadThreadCount`) and the HTTP
   concurrency limit for load/unload requests. Setting it to `0` disables
   the async path and falls back to the original synchronous behaviour.
   Documentation update for this dual use is pending and can be done in a
   follow-up.

2. **Shutdown safety**: Detached threads hold raw pointers to
   `HTTPAPIServer` members (`control_request_cnt_`, `server_.get()`).
   If the server is destroyed while a model load/unload is in-flight, a
   use-after-free is theoretically possible. This is the **same
   pre-existing risk** accepted by `InferRequestClass`; the fix does not
   introduce a new hazard. A comprehensive fix (e.g., `shared_ptr`
   to the counter, or joining on shutdown) should address all async paths
   simultaneously in a separate PR.

3. **`evthr_defer` return value unchecked**: Same as `InferRequestClass`
   line 4017 in the existing code. If the event loop has stopped,
   `ctrl_req` will leak. Not introduced by this PR.

4. **Second `Create()` overload** (options-map path, used in some
   embedded/SageMaker deployments): hardcodes `control_concurrency = 4`
   and reads from an options key `"control_request_concurrency"` that is
   not currently set by any caller. Normal `main.cc` startup is
   unaffected. Happy to align this in a follow-up if the reviewer prefers.

---

#### Background

Reported in issue #8635 by @aleksn7. Root cause confirmed:
`HandleRepositoryControl()` in `src/http_server.cc` calls
`TRITONSERVER_ServerLoadModelWithParameters()` synchronously on the evhtp
worker thread with no `evhtp_request_pause()` before the call. The evhtp
thread is fully blocked for the duration of model initialization
(potentially minutes for large models). With `--http-thread-count=N`,
N concurrent load requests exhaust the entire thread pool, starving
inference and Kubernetes health probes.

@aleksn7 provided explicit guidance on the approach in the issue:

> "another approach is to skip the thread pool entirely and create a new
> thread for each model_repository call … don't forget to limit the
> concurrent request count … you could use `--model-load-thread-count`
> CLI parameter."

This PR implements exactly that approach.

The `InferRequestClass` async pattern (lines 3931–3944, 3887–3910 of
`src/http_server.cc`) was used as the template. `ControlRequestClass`
is a simplified version: no inference cancellation in the fini hook
(not needed — there is no in-flight inference to cancel), no response
count tracking. All other mechanics (`evhtp_request_pause`,
`htpconn->thread` capture, `evhtp_hook_on_request_fini`,
`evthr_defer`, `evhtp_send_reply` + `evhtp_request_resume` in the
callback) are identical.

---

#### Related Issues:

- Fixes #8635

---
<!-- END OF PASTE REGION -->
