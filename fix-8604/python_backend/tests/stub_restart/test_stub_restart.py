# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Integration tests for Python backend stub restart (Issue #8604).

Tests verify that:
1. Normal inference works
2. Killing the stub triggers automatic restart
3. Inference succeeds after restart
4. Model readiness recovers after restart
5. Restart rate limiting prevents restart storms

These tests require a running Triton server with the kill_stub model loaded.
The model kills its stub process when INPUT[0] == 0.
"""

import sys
import time
import unittest

import numpy as np

sys.path.append("../../common")

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class TestStubRestart(unittest.TestCase):
    """Test stub restart functionality for Issue #8604."""

    MODEL_NAME = "kill_stub"
    # Maximum time to wait for stub restart (seconds)
    RESTART_TIMEOUT = 15
    # Poll interval when waiting for restart (seconds)
    POLL_INTERVAL = 0.5

    def setUp(self):
        self.http_client = httpclient.InferenceServerClient("localhost:8000")
        self.grpc_client = grpcclient.InferenceServerClient("localhost:8001")

    def tearDown(self):
        self.http_client.close()
        self.grpc_client.close()

    def _http_infer(self, value):
        """Send a single inference request via HTTP."""
        inp = httpclient.InferInput("INPUT", [1], "INT32")
        inp.set_data_from_numpy(np.array([value], dtype=np.int32))
        result = self.http_client.infer(self.MODEL_NAME, [inp])
        return result.as_numpy("OUTPUT")

    def _grpc_infer(self, value):
        """Send a single inference request via gRPC."""
        inp = grpcclient.InferInput("INPUT", [1], "INT32")
        inp.set_data_from_numpy(np.array([value], dtype=np.int32))
        result = self.grpc_client.infer(self.MODEL_NAME, [inp])
        return result.as_numpy("OUTPUT")

    def _wait_for_model_ready(self, timeout=None):
        """Poll until the model is ready or timeout expires.

        Uses polling instead of sleep to avoid flakiness.
        Returns True if model became ready, False if timed out.
        """
        if timeout is None:
            timeout = self.RESTART_TIMEOUT
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.http_client.is_model_ready(self.MODEL_NAME):
                    return True
            except Exception:
                pass
            time.sleep(self.POLL_INTERVAL)
        return False

    def _wait_for_model_not_ready(self, timeout=5):
        """Poll until the model is NOT ready or timeout expires.

        Returns True if model became not-ready, False if timed out.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if not self.http_client.is_model_ready(self.MODEL_NAME):
                    return True
            except Exception:
                return True
            time.sleep(self.POLL_INTERVAL)
        return False

    # ----- Test 1: Normal inference -----

    def test_normal_inference(self):
        """Verify that normal inference works with HTTP and gRPC."""
        # HTTP
        output = self._http_infer(42)
        self.assertEqual(output[0], 43, "HTTP inference returned wrong value")

        # gRPC
        output = self._grpc_infer(42)
        self.assertEqual(output[0], 43, "gRPC inference returned wrong value")

    # ----- Test 2: Restart after kill -----

    def test_restart_after_kill(self):
        """Verify that killing the stub triggers restart and subsequent
        inference succeeds.

        Sequence:
        1. Normal inference succeeds
        2. Kill-stub inference (INPUT=0) causes os._exit(0)
        3. The triggering request fails (expected)
        4. Wait for stub restart (poll model readiness)
        5. Next inference succeeds
        """
        # Step 1: Normal inference works
        output = self._http_infer(1)
        self.assertEqual(output[0], 2)

        # Step 2: Kill the stub
        with self.assertRaises(Exception) as ctx:
            self._http_infer(0)
        # The error should indicate the stub is unhealthy
        error_msg = str(ctx.exception)
        self.assertTrue(
            "not healthy" in error_msg.lower()
            or "internal" in error_msg.lower()
            or "unavailable" in error_msg.lower(),
            f"Unexpected error message: {error_msg}",
        )

        # Step 3: Wait for restart (poll, no sleep)
        ready = self._wait_for_model_ready()
        self.assertTrue(
            ready,
            f"Model did not become ready within {self.RESTART_TIMEOUT}s "
            "after stub restart",
        )

        # Step 4: Inference succeeds after restart
        output = self._http_infer(99)
        self.assertEqual(
            output[0], 100, "Inference after restart returned wrong value"
        )

        # Also verify via gRPC
        output = self._grpc_infer(99)
        self.assertEqual(
            output[0], 100, "gRPC inference after restart returned wrong value"
        )

    # ----- Test 3: Readiness after restart -----

    def test_readiness_after_restart(self):
        """Verify that model readiness correctly recovers after restart.

        Note: Per-model readiness (/v2/models/{model}/ready) already works
        correctly in upstream (via TRITONBACKEND_ModelInstanceReady ->
        StubActive()). This test verifies it continues to work after restart.
        """
        # Kill the stub
        try:
            self._http_infer(0)
        except Exception:
            pass  # Expected failure

        # Wait for restart
        ready = self._wait_for_model_ready()
        self.assertTrue(ready, "Model not ready after restart")

        # Verify via both HTTP and gRPC
        self.assertTrue(self.http_client.is_model_ready(self.MODEL_NAME))
        self.assertTrue(self.grpc_client.is_model_ready(self.MODEL_NAME))

    # ----- Test 4: Rate limiting -----

    def test_restart_rate_limit(self):
        """Verify that restart rate limiting prevents restart storms.

        The default rate limit is kMaxRestartsPerWindow=5 per
        kRestartWindowSeconds=60. After exceeding the limit, the model
        should remain unhealthy.

        We kill the stub repeatedly (more than the limit), then verify
        that the model stays not-ready.
        """
        # Kill the stub and wait for restart, repeat until we exceed the limit.
        # kMaxRestartsPerWindow = 5, so we need 6+ kills.
        for i in range(7):
            # Wait for model to become ready (from previous restart)
            ready = self._wait_for_model_ready(timeout=10)
            if not ready:
                # Model is not ready — rate limit may have been hit
                break

            # Kill the stub again
            try:
                self._http_infer(0)
            except Exception:
                pass  # Expected

            # Brief pause to let restart attempt happen
            time.sleep(1)

        # After exceeding rate limit, model should stay unhealthy
        # Wait a bit to ensure no more restarts happen
        time.sleep(3)
        model_ready = self.http_client.is_model_ready(self.MODEL_NAME)
        self.assertFalse(
            model_ready,
            "Model should be permanently unhealthy after exceeding "
            "restart rate limit",
        )


if __name__ == "__main__":
    unittest.main()
