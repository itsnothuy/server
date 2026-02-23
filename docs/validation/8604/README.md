# Validation Report: Issue #8604 — Python Backend Stub Death Invisible to Health API

> **Factified 2026-02-23** — Every claim re-verified against upstream code. See
> inline `[VERIFIED]`, `[CORRECTED]`, `[NOT VERIFIABLE]` tags.

## Scope

This validation assesses two ChatGPT agent-mode result files ("Python backend fix 1.txt" and "Python backend fix 2.txt") that propose fixes for [triton-inference-server/server#8604](https://github.com/triton-inference-server/server/issues/8604). It also performs a deep comparison against [python_backend PR #431](https://github.com/triton-inference-server/python_backend/pull/431), an existing open fix attempt by `@paipeline`.

## Environment

| Item | Value |
|------|-------|
| Validation date | 2026-02-23 (initial), 2026-02-23 (factified) |
| Validator | Claude Opus 4.6 (acting as senior C++/systems auditor) |
| Server repo SHA | `22fd79ea3271fe64e34746995986dc046ad02af1` (local `main`) |
| python_backend repo | Validated via GitHub API (upstream `main` branch, Feb 2026). Key files: `src/python_be.cc` (IsStubProcessAlive L155-171, SendMessageToStub L1072-1104, TRITONBACKEND_ModelInstanceReady L2418-2437, TRITONBACKEND_ModelInstanceExecute L2324-2416, LaunchStubProcess L326-346, destructor L1768-1781), `src/stub_launcher.cc` (StubActive L740-769). |
| PR #431 commit | `019296ff4ec721d3957088a5b7689f5e9478c4b3` (1 commit, 4 files, +300/-4) |
| Triton core repo | Validated via GitHub API. Key files: `src/server.cc` (IsReady L417-457, ModelIsReady L459-487), `src/backend_model.cc` (TritonModel::IsReady L295-306), `src/backend_model_instance.cc` (TritonModelInstance::IsReady L594-617). |
| Runtime execution | **NOT performed** — no Docker/GPU environment available |

## Files Validated

| File | Path | Size | Fully Read? |
|------|------|------|-------------|
| Python backend fix 1.txt | `/server/Python backend fix 1.txt` | 173 lines, 20148 bytes | ✅ Yes |
| Python backend fix 2.txt | `/server/Python backend fix 2.txt` | 173 lines, 20148 bytes | ✅ Yes |

**Note:** Both files are byte-for-byte **identical**. They contain the same analysis, repro, root-cause, and proposed fix.

## What Was Validated

- ✅ All factual claims in fix 1/2 against actual upstream code (python_backend `main`, triton core `main`)
- ✅ PR #431 diff (complete `.diff` fetched), claims, code quality, and test coverage
- ✅ Server-side readiness architecture (`IsReady`, `ModelIsReady`, `TRITONBACKEND_ModelInstanceReady`) — full call chains traced
- ✅ Existing upstream tests (`qa/L0_backend_python/model_readiness/test.sh` — complete content reviewed)
- ✅ `strict_readiness` flag semantics (default=`true`, confirmed in `InferenceServer` constructor)

## What Was NOT Validated

- ❌ **Runtime reproduction** — no Docker or GPU environment available to run Triton containers
- ❌ **Build validation** — cannot compile python_backend or server locally
- ❌ **r24.05 branch code** — validated claims about old restart logic based on fix file descriptions and upstream code inspection, not by checking out the old branch. The TODO comment presence + absence of restart code is consistent with the claim.
- ❌ **PR #431 CI results** — PR shows 0 reviews, 0 CI checks (public page only)

## Report Index

1. [our-approach-summary.md](our-approach-summary.md) — Summary of fix 1/2 content
2. [claim-matrix.md](claim-matrix.md) — Claim-by-claim validation (18 claims; 10 ✅, 2 ❌, 4 ⚠️, 2 ❓)
3. [pr431-review.md](pr431-review.md) — PR #431 deep review (6 critical flaws identified)
4. [repro-validation.md](repro-validation.md) — Reproduction constraints
5. [recommendations.md](recommendations.md) — Final recommendation
6. [pr-plan.md](pr-plan.md) — Concrete PR and test plan

## Key Corrections from Factification

1. **[CORRECTED] Claim 2.3**: The fix files claim `TRITONBACKEND_ModelInstanceReady` is "cached/only called at startup". This is **false**. It is called dynamically on every `/v2/models/{model}/ready` request. Evidence: `TritonModelInstance::IsReady()` (core L594-617) calls `TRITONBACKEND_ModelInstanceReady` function pointer on every invocation. No caching.
2. **[CORRECTED] Claim 2.4**: The fix files conflate per-model readiness with server-level readiness. `/v2/models/{model}/ready` **already works** (detects dead stubs). Only `/v2/health/ready` is broken.
3. **[CORRECTED] Claim 2.2**: There is no `restart` variable on `main`. The TODO comment is orphaned — it references a concept that was fully removed.
4. **[CORRECTED] Claim 3.4**: `strict_readiness_` defaults to `true`, not `false` as the fix files imply.
5. **[CORRECTED] PR #431 test validation**: Tests are grep-based static checks on source code, not integration tests. They provide zero behavioral confidence.
