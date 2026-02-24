# Track A vs Track B — Executive Summary

> **Source:** Claude Opus 4.6 analysis, 2026-02-24  
> **Context:** Side-by-side comparison of two proposed fixes for `triton-inference-server` Issue #8604

---

Track B is the correct fix for Issue #8604. Track A is a valuable enhancement but is not the fix — it is new functionality that masks the root cause while introducing significant complexity. Track B surgically corrects the one broken code path (`InferenceServer::IsReady()` not calling `ModelIsReady()`) with ~15 lines, zero inference-path impact, and zero new failure modes. Track A adds ~120 lines of production code with a new mutex, atomic state, rate limiting, and process lifecycle management in the inference hot path — a contribution that CONTRIBUTING.md explicitly says requires prior design discussion with maintainers. Submit Track B as a PR now. Post Track A as a design proposal on Issue #8604 and wait for maintainer feedback before writing the PR.
