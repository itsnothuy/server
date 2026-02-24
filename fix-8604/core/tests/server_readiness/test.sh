#!/bin/bash
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# =============================================================================
# Integration test for /v2/health/ready with --strict-readiness=true
# when a Python backend stub dies (Issue #8604, Track B).
#
# This test verifies that server-level readiness correctly reflects
# backend runtime health, not just lifecycle state.
#
# Prerequisites:
# - Triton server built with the core readiness fix
# - Python backend available
# - tritonclient[all] installed
# =============================================================================

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    exit 1
fi

source ../common/util.sh

RET=0
TEST_LOG="./server_readiness_test.log"
SERVER_LOG="./server_readiness_server.log"
MODELDIR="${PWD}/models"

# Clean up from previous runs
rm -rf ${MODELDIR} ${TEST_LOG} ${SERVER_LOG}

# ---- Setup: Create the identity model (used for readiness) ----
mkdir -p ${MODELDIR}/identity_fp32/1

cat > ${MODELDIR}/identity_fp32/config.pbtxt << 'EOF'
name: "identity_fp32"
backend: "python"
max_batch_size: 0
input [{
  name: "INPUT0"
  data_type: TYPE_FP32
  dims: [1]
}]
output [{
  name: "OUTPUT0"
  data_type: TYPE_FP32
  dims: [1]
}]
instance_group [{
  count: 1
  kind: KIND_CPU
}]
EOF

cat > ${MODELDIR}/identity_fp32/1/model.py << 'PYEOF'
import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_tensor])
            )
        return responses
PYEOF

# ---- Start Triton with --strict-readiness=true (the default, but explicit) ----
SERVER_ARGS="--model-repository=${MODELDIR} --backend-directory=${BACKEND_DIR} --strict-readiness=true --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start server\n***"
    cat ${SERVER_LOG}
    exit 1
fi

# ---- Test 1: Server is ready with healthy model ----
set +e
python3 test_server_readiness.py TestServerReadiness.test_server_ready_initially >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test 1 FAILED: Server should be ready initially\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

# ---- Test 2: Kill stub, verify server readiness reflects it ----
# Find the stub PID and kill it
STUB_PID=$(pgrep -f "triton_python_backend_stub.*identity_fp32")
if [ -z "$STUB_PID" ]; then
    echo -e "\n***\n*** FAILED: Could not find stub process\n***"
    RET=1
else
    echo "Killing stub PID: $STUB_PID"
    kill -9 $STUB_PID

    set +e
    python3 test_server_readiness.py TestServerReadiness.test_server_not_ready_after_stub_death >>$TEST_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test 2 FAILED: Server should NOT be ready after stub death\n***"
        cat $TEST_LOG
        RET=1
    fi
    set -e
fi

# ---- Test 3: Per-model readiness still works (baseline, should always pass) ----
set +e
python3 test_server_readiness.py TestServerReadiness.test_model_not_ready_after_stub_death >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test 3 FAILED: Per-model readiness check\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

kill_server

# ---- Test 4: Server readiness with strict=false (should remain ready) ----
rm -f ${SERVER_LOG}

# Restart with a fresh stub for strict=false test
SERVER_ARGS="--model-repository=${MODELDIR} --backend-directory=${BACKEND_DIR} --strict-readiness=false --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start server for strict=false test\n***"
    cat ${SERVER_LOG}
    exit 1
fi

# Kill stub again
STUB_PID=$(pgrep -f "triton_python_backend_stub.*identity_fp32")
if [ -z "$STUB_PID" ]; then
    echo -e "\n***\n*** FAILED: Could not find stub process (strict=false test)\n***"
    RET=1
else
    kill -9 $STUB_PID

    set +e
    python3 test_server_readiness.py TestServerReadiness.test_server_ready_with_strict_false >>$TEST_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test 4 FAILED: Server should be ready with strict=false\n***"
        cat $TEST_LOG
        RET=1
    fi
    set -e
fi

kill_server

# ---- Summary ----
if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
    cat $TEST_LOG
fi

exit $RET
