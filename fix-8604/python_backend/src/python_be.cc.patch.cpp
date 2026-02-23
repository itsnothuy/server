// =============================================================================
// Fix for triton-inference-server/server#8604
// File: python_backend/src/python_be.cc — RestartStubProcess() implementation
//
// This file contains the production-ready RestartStubProcess() method and
// the modified TRITONBACKEND_ModelInstanceExecute() with restart detection.
//
// To apply: These functions replace/augment the existing code in python_be.cc.
// See the inline comments for exact insertion points.
// =============================================================================

// ---------------------------------------------------------------------------
// NEW METHOD: ModelInstanceState::CanRestart()
// Insert after ModelInstanceState::LaunchStubProcess() (around line 346)
// ---------------------------------------------------------------------------

bool
ModelInstanceState::CanRestart()
{
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                     now - restart_window_start_)
                     .count();
  if (elapsed > kRestartWindowSeconds) {
    // Reset the sliding window
    restart_count_ = 0;
    restart_window_start_ = now;
  }
  return restart_count_ < kMaxRestartsPerWindow;
}

// ---------------------------------------------------------------------------
// NEW METHOD: ModelInstanceState::RestartStubProcess()
// Insert after CanRestart()
// ---------------------------------------------------------------------------

TRITONSERVER_Error*
ModelInstanceState::RestartStubProcess()
{
  // This method is always called under restart_mutex_ protection.
  // See TRITONBACKEND_ModelInstanceExecute for the locking pattern.

  if (!CanRestart()) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        (std::string("Stub restart rate limit exceeded for instance '") +
         Name() + "' (" + std::to_string(restart_count_) + " restarts in " +
         std::to_string(kRestartWindowSeconds) +
         "s window). Instance is permanently unhealthy.")
            .c_str());
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("Stub restart rate limit exceeded for instance '") +
         Name() + "'. Instance is permanently unhealthy.")
            .c_str());
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Restarting unhealthy stub process for instance '") +
       Name() + "' (restart #" + std::to_string(restart_count_ + 1) + ")")
          .c_str());

  try {
    if (Stub()) {
      // IMPORTANT: Do NOT call UpdateHealth()/IsHealthy() here.
      // The stub is dead or unresponsive. UpdateHealth() tries to acquire
      // the health mutex, which may be held by the dead stub process,
      // causing a 1-second block. IsHealthy() would be unreliable.
      //
      // Do NOT call thread_pool_->wait() here.
      // Pending tasks may be stuck on IPC operations (SendMessageToStub,
      // ReceiveMessageFromStub) that will never complete because the stub
      // is dead. Waiting would block indefinitely.

      // Kill the stub process first (TerminateStub sends SIGKILL if
      // graceful finalize fails, which it will since the stub is dead).
      Stub()->TerminateStub();

      // Stop the monitor thread. It may be blocked on the parent message
      // queue; TerminateStub should unblock it by cleaning up the stub side.
      TerminateMonitor();

      // Drain any remaining messages from shared memory queues.
      Stub()->ClearQueues();

      // Destroy the old StubLauncher and all its shared memory regions.
      Stub().reset();
    }

    // LaunchStubProcess creates a completely new StubLauncher with fresh
    // shared memory, starts the monitor thread, and creates new
    // thread_pool_ and request_executor_ objects.
    RETURN_IF_ERROR(LaunchStubProcess());

    restart_count_++;

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Successfully restarted stub process for instance '") +
         Name() + "'")
            .c_str());

    return nullptr;  // success
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("Failed to restart stub process for instance '") +
         Name() + "': " + ex.what())
            .c_str());
  }
}

// ---------------------------------------------------------------------------
// MODIFIED FUNCTION: TRITONBACKEND_ModelInstanceExecute
// This replaces the existing function at ~line 2324
//
// Changes from upstream main:
// 1. Removed orphaned TODO comment about restart
// 2. Added restart detection after ProcessRequests() error
// 3. Uses IsStubProcessAlive() for detection (not string matching)
// 4. Guarded by restart_mutex_ + restart_in_progress_ atomic
// 5. Does NOT retry current requests (fails them, restarts for next)
// ---------------------------------------------------------------------------

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  TRITONSERVER_Error* error = nullptr;

  // NOTE: The old TODO comment "// TODO: Implement restart on decoupled"
  // has been removed. Restart is now implemented below for both decoupled
  // and non-decoupled models.

  std::vector<std::unique_ptr<InferRequest>> infer_requests;
  {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    PbMetricReporter reporter(
        instance_state->TritonModelInstance(), requests, request_count,
        nullptr);
    reporter.SetExecStartNs(exec_start_ns);

    error = instance_state->ProcessRequests(
        requests, request_count, infer_requests, reporter);

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    reporter.SetExecEndNs(exec_end_ns);

    if (error != nullptr) {
      reporter.SetSuccessStatus(false);

      // --- BEGIN RESTART DETECTION ---
      // Check if the stub process is dead. We use IsStubProcessAlive()
      // instead of string matching on error messages. This catches ALL
      // failure modes:
      //   - "Stub process is not healthy." (from SendMessageToStub)
      //   - "Failed to obtain the health mutex." (from SendMessageToStub)
      //   - Any ReceiveMessageFromStub failure when stub is dead
      //
      // We do NOT retry the current requests after restart because:
      //   1. ProcessRequests may have partially consumed request resources
      //      (response factories, shared memory allocations)
      //   2. Retrying with the same request pointers risks double-free
      //   3. The PbMetricReporter has already recorded timing for this batch
      //
      // Instead, we fail the current batch (normal error handling below)
      // and restart the stub so subsequent requests succeed.
      if (!instance_state->IsStubProcessAlive()) {
        // Try to acquire restart mutex without blocking.
        // If another thread is already restarting, we just fail our batch.
        std::unique_lock<std::mutex> restart_lock(
            instance_state->restart_mutex_, std::try_to_lock);
        if (restart_lock.owns_lock() &&
            !instance_state->restart_in_progress_.exchange(true)) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_WARN,
              (std::string("Detected dead stub process for instance '") +
               instance_state->Name() + "', attempting restart")
                  .c_str());

          TRITONSERVER_Error* restart_error =
              instance_state->RestartStubProcess();
          instance_state->restart_in_progress_ = false;

          if (restart_error != nullptr) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                (std::string("Stub restart failed for instance '") +
                 instance_state->Name() + "': " +
                 TRITONSERVER_ErrorMessage(restart_error))
                    .c_str());
            TRITONSERVER_ErrorDelete(restart_error);
          }
          // Note: 'error' from ProcessRequests is preserved.
          // The current batch will still be failed with the original error.
        }
      }
      // --- END RESTART DETECTION ---

      // Original error handling: send error responses for all requests
      for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];
        if (!instance_state->ExistsInClosedRequests(
                reinterpret_cast<intptr_t>(request))) {
          TRITONBACKEND_Response* response = nullptr;
          LOG_IF_ERROR(
              TRITONBACKEND_ResponseNew(&response, request),
              "Failed to create a new response.");
          if (response != nullptr) {
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSend(
                    response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
                "Failed to send the error response.");
          }
        }
      }

      for (auto& infer_request : infer_requests) {
        // Reset the release flags for all the requests.
        infer_request->SetReleaseFlags(TRITONSERVER_REQUEST_RELEASE_ALL);
      }
    }
  }

  // The InferRequest object might not be created if an error occurs. Explicitly
  // update the release flags here based on the number of InferRequest objects.
  std::vector<uint32_t> request_release_flags(
      request_count, TRITONSERVER_REQUEST_RELEASE_ALL);
  for (size_t i = 0; i < infer_requests.size(); ++i) {
    request_release_flags[i] = infer_requests[i]->ReleaseFlags();
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    try {
      THROW_IF_TRITON_ERROR(
          TRITONBACKEND_RequestRelease(request, request_release_flags[r]));
    }
    catch (const PythonBackendException& pb_exception) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Failed to release request: ") + pb_exception.what())
              .c_str());
      // If RESCHEDULE fails, release with ALL to avoid leak
      if (request_release_flags[r] ==
          TRITONSERVER_REQUEST_RELEASE_RESCHEDULE) {
        LOG_IF_ERROR(
            TRITONBACKEND_RequestRelease(
                request, TRITONSERVER_REQUEST_RELEASE_ALL),
            "Failed to release request after reschedule failure.");
      }
    }
  }

  TRITONSERVER_ErrorDelete(error);
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceExecute: model instance name ") +
       instance_state->Name() + " released " +
       std::to_string(request_count) + " requests")
          .c_str());

  return nullptr;  // success
}
