# Our Approach Summary

## Source Files

- `Python backend fix 1.txt` (173 lines, 20148 bytes)
- `Python backend fix 2.txt` (173 lines, 20148 bytes)

**Important finding:** Both files are **byte-for-byte identical**. There is only one unique analysis document, duplicated across two files. All quotes below apply to both.

## High-Level Summary

The document is a comprehensive analysis of Triton Inference Server issue #8604, structured as:

1. **Executive summary** — Problem statement about Python backend stub process death being invisible to health endpoints
2. **Reproducible test case** — Docker-based repro using `os._exit(0)` to kill the stub
3. **Root-cause analysis** — Detailed code-level walkthrough of the failure
4. **Proposed fixes** — Two complementary tracks (A: stub restart, B: readiness propagation)
5. **Regression tests** — Test plan for both tracks
6. **Patch plan** — Implementation steps across python_backend and server repos
7. **Operational recommendations** — Short-term workarounds

## Key Claims Made

### Architecture Claims

> "The Python backend isolates a user-provided model.py in a separate stub process. The Triton core communicates with this stub using shared-memory message queues and a health mutex to detect when the stub process is stuck or has exited."

> "When the parent wants to send a command to the stub it calls ModelInstanceState::SendMessageToStub. This function acquires the health mutex with a 1-second timeout, sets stub_health to false and pushes the message onto the IPC queue."

> "ModelInstanceState::IsStubProcessAlive tries to lock the health mutex for one second. If it acquires the lock it returns the value of stub_health; if the lock cannot be obtained it assumes the stub has died or is blocked."

### Root-Cause Claims

> "A refactor to unify decoupled and non-decoupled execution (python backend PR #360) removed the restart block and replaced it with a TODO to implement restart for decoupled models."

> "After PR #360 unified decoupled and non-decoupled pipelines, the above restart block was removed and replaced with a TODO comment: 'If restart is equal to true, it indicates that the stub process is unhealthy and needs a restart. // TODO: Implement restart on decoupled'"

> "TRITONBACKEND_ModelInstanceReady is called only during model loading. It checks if the stub is active and returns an error if not. There is no periodic check after initialization."

> "The health API in Triton's core sets the server as ready when models are loaded; it doesn't re-invoke ModelInstanceReady on every health request."

### Proposed Fix — Track A (Stub Restart)

The document proposes re-introducing restart logic with:
- Mutex to prevent concurrent requests during restart
- TerminateMonitor() → KillStubProcess() → Setup() → StartMonitor() → Launch() sequence
- Decoupled model handling (close in-flight streams)
- Retry/backoff strategy (`--python-stub-max-restarts`, `--python-stub-restart-window-sec`)
- Thread safety via atomic pointer
- `--python-disable-stub-restart` flag

### Proposed Fix — Track B (Readiness Propagation)

The document proposes:
- `instance_unhealthy` flag in `ModelInstanceState`
- Modified `TRITONBACKEND_ModelInstanceReady` to check the flag
- New `CheckRuntimeModelReadiness()` function in Triton server core
- Modified `/v2/health/ready` to call this function under `--strict-readiness=true`
- Backwards compatibility under default `strict-readiness=false`

### Test Plan

- Backend restart test: kill stub → verify restart → verify subsequent inference succeeds
- Readiness propagation test: kill stub → verify health endpoints return non-200

### Referenced Issues

- [#8604](https://github.com/triton-inference-server/server/issues/8604) — Primary issue
- [#7230](https://github.com/triton-inference-server/server/issues/7230) — Related zombie stub issue
- [#8518](https://github.com/triton-inference-server/server/issues/8518) — Related (mentioned but not cited with URL)
- [PR #360](https://github.com/triton-inference-server/python_backend/pull/360) — Pipeline unification PR
