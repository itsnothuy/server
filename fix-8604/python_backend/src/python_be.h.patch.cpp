// =============================================================================
// Fix for triton-inference-server/server#8604
// File: python_backend/src/python_be.h — Header additions
//
// Add these declarations to the ModelInstanceState class.
// =============================================================================
//
// --- PUBLIC SECTION (after LaunchStubProcess declaration) ---
//
// Add these method declarations:

  /// Restart an unhealthy stub process with rate limiting.
  /// Thread-safe: callers use restart_mutex_ externally.
  TRITONSERVER_Error* RestartStubProcess();

  /// Check whether a restart is permitted by rate limiter.
  bool CanRestart();

  /// Check if the stub process is currently alive.
  /// Already exists in upstream — listed here for completeness.
  // bool IsStubProcessAlive();  // already declared

//
// --- PRIVATE SECTION (add new members) ---
//
// Add these member variables:

  // Concurrency protection for restart path.
  // Used with try_to_lock in TRITONBACKEND_ModelInstanceExecute to prevent
  // multiple threads from attempting restart simultaneously.
  std::mutex restart_mutex_;
  std::atomic<bool> restart_in_progress_{false};

  // Restart rate limiting.
  // Prevents restart storms when model code is fundamentally broken.
  // After kMaxRestartsPerWindow restarts within kRestartWindowSeconds,
  // the instance is marked permanently unhealthy.
  std::atomic<int> restart_count_{0};
  std::chrono::steady_clock::time_point restart_window_start_{
      std::chrono::steady_clock::now()};
  static constexpr int kMaxRestartsPerWindow = 5;
  static constexpr int kRestartWindowSeconds = 60;

//
// --- REQUIRED INCLUDES ---
//
// Ensure these are included at the top of python_be.h (most already are):
// #include <atomic>
// #include <chrono>
// #include <mutex>
