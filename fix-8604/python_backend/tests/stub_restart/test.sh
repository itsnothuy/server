#!/bin/bash
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

# =============================================================================
# Integration test for python_backend stub restart (Issue #8604)
#
# Tests:
# 1. Normal inference succeeds
# 2. Killing stub via os._exit(0) triggers automatic restart
# 3. Inference succeeds after restart
# 4. Model readiness recovers after restart
# 5. Restart rate limiting works (stub stays unhealthy after too many restarts)
#
# Prerequisites:
# - Triton server with python backend built with restart fix
# - tritonclient[all] installed
# - This test follows the pattern from qa/L0_backend_python/model_readiness/
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
TEST_LOG="./stub_restart_test.log"
SERVER_LOG="./stub_restart_server.log"
MODELDIR="${PWD}/models"

# Clean up from previous runs
rm -rf ${MODELDIR} ${TEST_LOG} ${SERVER_LOG}

# ---- Setup: Create the kill_stub model ----
mkdir -p ${MODELDIR}/kill_stub/1

cat > ${MODELDIR}/kill_stub/config.pbtxt << 'EOF'
name: "kill_stub"
backend: "python"
max_batch_size: 0
input [{
  name: "INPUT"
  data_type: TYPE_INT32
  dims: [1]
}]
output [{
  name: "OUTPUT"
  data_type: TYPE_INT32
  dims: [1]
}]
instance_group [{
  count: 1
  kind: KIND_CPU
}]
EOF

cat > ${MODELDIR}/kill_stub/1/model.py << 'PYEOF'
import triton_python_backend_utils as pb_utils
import numpy as np
import os

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()
            if inp[0] == 0:
                # Kill the stub process. This simulates issue #8604.
                os._exit(0)
            out_tensor = pb_utils.Tensor("OUTPUT", inp + 1)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_tensor])
            )
        return responses

    def finalize(self):
        pass
PYEOF

# ---- Start Triton ----
SERVER_ARGS="--model-repository=${MODELDIR} --backend-directory=${BACKEND_DIR} --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start server\n***"
    cat ${SERVER_LOG}
    exit 1
fi

# ---- Test 1: Normal inference succeeds ----
set +e
python3 test_stub_restart.py TestStubRestart.test_normal_inference >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test 1 FAILED: Normal inference\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

# ---- Test 2: Restart after kill ----
set +e
python3 test_stub_restart.py TestStubRestart.test_restart_after_kill >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test 2 FAILED: Restart after kill\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

# ---- Verify restart occurred in server log ----
RESTART_COUNT=$(grep -c "Successfully restarted stub process for instance 'kill_stub_0_0'" ${SERVER_LOG})
if [ "$RESTART_COUNT" -lt 1 ]; then
    echo -e "\n***\n*** FAILED: Expected at least 1 restart log entry, got ${RESTART_COUNT}\n***"
    RET=1
fi

DETECTION_COUNT=$(grep -c "Detected dead stub process for instance 'kill_stub_0_0'" ${SERVER_LOG})
if [ "$DETECTION_COUNT" -lt 1 ]; then
    echo -e "\n***\n*** FAILED: Expected at least 1 detection log entry, got ${DETECTION_COUNT}\n***"
    RET=1
fi

# ---- Test 3: Model readiness after restart ----
set +e
python3 test_stub_restart.py TestStubRestart.test_readiness_after_restart >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test 3 FAILED: Readiness after restart\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

kill_server

# ---- Test 4: Rate limiting (separate server run to start clean) ----
rm -f ${SERVER_LOG}
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start server for rate limit test\n***"
    cat ${SERVER_LOG}
    exit 1
fi

set +e
python3 test_stub_restart.py TestStubRestart.test_restart_rate_limit >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test 4 FAILED: Rate limiting\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

# Verify rate limit message in server log
RATE_LIMIT_COUNT=$(grep -c "Stub restart rate limit exceeded" ${SERVER_LOG})
if [ "$RATE_LIMIT_COUNT" -lt 1 ]; then
    echo -e "\n***\n*** FAILED: Expected rate limit exceeded log entry\n***"
    RET=1
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
