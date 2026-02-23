#!/usr/bin/env python3
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
Integration tests for /v2/health/ready with dead Python backend stubs.

These tests verify the fix for Issue #8604: when a Python backend stub
process dies, the server-level readiness endpoint (/v2/health/ready)
should report NOT READY (with --strict-readiness=true).

Previously, /v2/health/ready only checked ModelStates() (lifecycle state),
which remains READY even when the underlying stub process is dead.
The fix adds a ModelIsReady() call that invokes TRITONBACKEND_ModelInstanceReady,
which detects the dead stub via IsStubProcessAlive().

Test architecture:
  - test_server_ready_initially:         /v2/health/ready returns 200
  - test_server_not_ready_after_stub_death: /v2/health/ready returns non-200
  - test_model_not_ready_after_stub_death:  /v2/models/.../ready returns non-200
  - test_server_ready_with_strict_false:    strict=false => always 200
"""

import sys
import time
import unittest

import requests


class TestServerReadiness(unittest.TestCase):
    """Test server-level readiness reflects backend health."""

    TRITON_URL = "http://localhost:8000"
    HEALTH_READY_URL = f"{TRITON_URL}/v2/health/ready"
    MODEL_READY_URL = f"{TRITON_URL}/v2/models/identity_fp32/ready"

    def test_server_ready_initially(self):
        """Server should be ready when all models are healthy."""
        resp = requests.get(self.HEALTH_READY_URL, timeout=5)
        self.assertEqual(
            resp.status_code,
            200,
            f"/v2/health/ready should return 200, got {resp.status_code}",
        )

    def test_server_not_ready_after_stub_death(self):
        """
        After stub death, /v2/health/ready should return non-200
        with --strict-readiness=true.

        This is the CORE test for Issue #8604.

        Before the fix: /v2/health/ready returns 200 (WRONG)
        After the fix:  /v2/health/ready returns 503  (CORRECT)
        """
        # Poll for up to 10 seconds for the server to detect the dead stub.
        # IsStubProcessAlive() uses a 1-second health mutex timeout,
        # so detection should happen within a few seconds.
        deadline = time.time() + 10
        last_status = None

        while time.time() < deadline:
            resp = requests.get(self.HEALTH_READY_URL, timeout=5)
            last_status = resp.status_code
            if last_status != 200:
                break
            time.sleep(0.5)

        self.assertNotEqual(
            last_status,
            200,
            "/v2/health/ready returned 200 after stub death — "
            "the core bug in Issue #8604 is NOT fixed. "
            "InferenceServer::IsReady() is not checking runtime backend health.",
        )

    def test_model_not_ready_after_stub_death(self):
        """
        Per-model readiness (/v2/models/.../ready) should return non-200
        after stub death. This endpoint already works correctly via
        ModelIsReady() → TRITONBACKEND_ModelInstanceReady → IsStubProcessAlive().

        This test serves as a baseline — if it fails, the backend health
        detection itself is broken, not just the server-level aggregation.
        """
        deadline = time.time() + 10
        last_status = None

        while time.time() < deadline:
            resp = requests.get(self.MODEL_READY_URL, timeout=5)
            last_status = resp.status_code
            if last_status != 200:
                break
            time.sleep(0.5)

        self.assertNotEqual(
            last_status,
            200,
            "/v2/models/identity_fp32/ready returned 200 after stub death — "
            "backend IsStubProcessAlive() detection is broken.",
        )

    def test_server_ready_with_strict_false(self):
        """
        With --strict-readiness=false, /v2/health/ready should return 200
        even when a model's stub is dead. strict-readiness=false means
        readiness only requires the server to have started successfully,
        not that all models are operational.
        """
        resp = requests.get(self.HEALTH_READY_URL, timeout=5)
        self.assertEqual(
            resp.status_code,
            200,
            f"/v2/health/ready should return 200 with strict=false, "
            f"got {resp.status_code}",
        )


if __name__ == "__main__":
    unittest.main()
