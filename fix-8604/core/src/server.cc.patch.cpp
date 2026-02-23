// =============================================================================
// Fix for triton-inference-server/server#8604 — Track B
// File: triton-inference-server/core/src/server.cc
//
// Target: InferenceServer::IsReady()
// Purpose: Under strict_readiness_=true, check runtime backend instance
//          readiness (not just lifecycle state) so /v2/health/ready correctly
//          reflects dead backends.
//
// This is a BUGFIX: The Triton documentation states that /v2/health/ready
// indicates "whether the server is able to respond to inference requests."
// With --strict-readiness=true, a model with a dead Python stub cannot
// respond to inference, yet IsReady() was returning true.
//
// Root cause: IsReady() only checks ModelStates() (lifecycle enum), while
// ModelIsReady() also calls model->IsReady() which invokes
// TRITONBACKEND_ModelInstanceReady — a live backend check.
//
// The fix adds a runtime readiness check for models in READY lifecycle state
// when strict_readiness_ is true.
//
// Performance note: TRITONBACKEND_ModelInstanceReady in python_backend calls
// StubActive() which does waitpid(WNOHANG) — a non-blocking, microsecond-level
// check. There is no meaningful overhead.
// =============================================================================

// ORIGINAL CODE (core/src/server.cc lines 417-457):
//
// Status
// InferenceServer::IsReady(bool* ready)
// {
//   *ready = false;
//
//   if (ready_state_ == ServerReadyState::SERVER_EXITING) {
//     return Status(Status::Code::UNAVAILABLE, "Server exiting");
//   }
//
//   ScopedAtomicIncrement inflight(inflight_request_counter_);
//
//   // Server is considered ready if it is in the ready state.
//   // Additionally can report ready only when all models are ready.
//   *ready = (ready_state_ == ServerReadyState::SERVER_READY);
//   if (*ready && strict_readiness_) {
//     // Strict readiness... get the model status and make sure all
//     // models are ready.
//     const auto model_versions = model_repository_manager_->ModelStates();
//
//     for (const auto& mv : model_versions) {
//       // If a model status is present but no version status,
//       // the model is not ready as there is no proper version to be served
//       if (mv.second.size() == 0) {
//         *ready = false;
//         goto strict_done;
//       }
//       for (const auto& vs : mv.second) {
//         // Okay if model is not ready due to unload
//         if ((vs.second.first != ModelReadyState::READY) &&
//             (vs.second.second != "unloaded")) {
//           *ready = false;
//           goto strict_done;
//         }
//       }
//     }
//   strict_done:;
//   }
//
//   return Status::Success;
// }

// PATCHED CODE:

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
