# Track B: Remaining Steps Before PR Submission

**Status:** Items 1-2 completed ✅. Items 3-6 require manual execution.

---

## ✅ Completed (Committed to `fix-8604-server-readiness`)

1. **Fixed copyright year** — `2026` → `2025` in both test files
2. **Removed `sleep 2`** — Tests 2 and 4 now rely solely on Python polling loops

---

## 🔨 Item 3: Compile Against Actual `triton-inference-server/core`

**Why this is required:** The patch exists as a `.patch.cpp` file that has never been compiled. You must verify it compiles cleanly in the actual Triton build system.

**Steps:**

### 3.1: Fork and Clone `triton-inference-server/core`

```bash
# 1. On GitHub, fork https://github.com/triton-inference-server/core
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/core.git triton-core
cd triton-core

# 3. Create a feature branch
git checkout -b fix-8604-server-readiness

# 4. Verify you're on the correct repo
git remote -v
# Should show YOUR_USERNAME/core, not itsnothuy/server
```

### 3.2: Apply the Patch to `src/server.cc`

```bash
# Copy the patched function from your server repo
# File: /Users/tranhuy/Desktop/Code/server/fix-8604/core/src/server.cc.patch.cpp
# Lines 70-140 contain the patched IsReady() function

# Open src/server.cc in the core repo and replace the IsReady() function
# (Lines 417-457 in upstream) with the patched version
```

**Manual edit required:**
- Open `triton-core/src/server.cc`
- Find `Status InferenceServer::IsReady(bool* ready)` (around line 417)
- Replace the entire function body with the patched version from `server.cc.patch.cpp`

### 3.3: Build the Core Repo

```bash
cd triton-core

# Install build dependencies (if not already done)
# See https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DTRITON_ENABLE_LOGGING=ON \
      -DTRITON_ENABLE_STATS=ON \
      -DTRITON_ENABLE_GPU=OFF \
      ..

# Build (this will take 10-30 minutes)
make -j$(nproc)
```

**Expected result:** Zero compile errors. If successful, you've verified the patch is syntactically correct.

**If you get errors:**
- "jump bypasses variable initialization" → The `goto strict_done` analysis was wrong (unlikely)
- "`name_` is not a member of..." → Type mismatch (extremely unlikely per validation)
- Any other error → Report it back for analysis

---

## 🧪 Item 4: Run Existing Tests (`L0_backend_python` minimum)

**Why:** Verify the patch doesn't break existing functionality.

**Prerequisites:**
- Compiled `core` repo from Item 3
- Triton `server` repo with the patched `core` as a submodule

**Option A: Full Triton Build (Recommended)**

```bash
# 1. Clone triton-inference-server/server
git clone https://github.com/triton-inference-server/server.git triton-server
cd triton-server

# 2. Update the core submodule to point to your patched fork
cd core
git remote add myfork https://github.com/YOUR_USERNAME/core.git
git fetch myfork
git checkout myfork/fix-8604-server-readiness
cd ..

# 3. Build the full server
python3 build.py --enable-logging --enable-stats --backend=python

# 4. Run L0_backend_python tests
cd qa/L0_backend_python
./test.sh
```

**Option B: Docker Container (Faster)**

```bash
# Pull the Triton SDK container
docker pull nvcr.io/nvidia/tritonserver:24.01-py3-sdk

# Mount your patched core and run tests
docker run -it --rm \
    -v /path/to/your/patched/core:/workspace/core \
    nvcr.io/nvidia/tritonserver:24.01-py3-sdk \
    bash -c "cd /workspace && ./qa/L0_backend_python/test.sh"
```

**What to verify:**
- All existing tests pass (especially `L0_backend_python/model_readiness`)
- No new errors or warnings in server logs
- Server starts and loads Python models successfully

---

## 🧪 Item 5: Run New Test on Live Patched Triton

**Why:** Verify the new test passes and actually reproduces the #8604 fix.

**Steps:**

### 5.1: Prepare the Test Environment

```bash
# From your triton-server build directory
cd qa

# Copy the new test
mkdir -p L0_server_readiness
cp /Users/tranhuy/Desktop/Code/server/fix-8604/core/tests/server_readiness/* \
   L0_server_readiness/
```

### 5.2: Install Python Dependencies

```bash
# In the container or your build environment
pip install requests  # or tritonclient[http] if you migrate the test
```

### 5.3: Run the Test

```bash
cd qa/L0_server_readiness

# Set environment variables
export NVIDIA_TRITON_SERVER_VERSION=24.01  # or your build version
export BACKEND_DIR=/path/to/triton/backends

# Run the test
./test.sh
```

**Expected output:**
```
Test 1 PASSED: Server ready initially
Test 2 PASSED: Server NOT ready after stub death (THE FIX)
Test 3 PASSED: Per-model readiness also detects dead stub
Test 4 PASSED: Server ready with strict=false

***
*** Test PASSED
***
```

**If Test 2 fails (returns 200 when it should return non-200):**
- The patch was not applied correctly
- The server binary you're testing is not the patched version
- The test itself has a bug

---

## 📝 Item 6: Sign NVIDIA CLA & Verify Target Repo

### 6.1: Sign the NVIDIA CLA

1. Go to https://github.com/triton-inference-server/core
2. Check if there's a CLA requirement (likely in CONTRIBUTING.md)
3. Follow the CLA signing process (usually involves agreeing to terms on first PR)

**Alternative:** Wait until you open the PR — GitHub will prompt you to sign if needed.

### 6.2: Verify PR Target

**Current state:**
- Your patch is on `itsnothuy/server` repo (wrong target)
- The PR must go to `triton-inference-server/core` repo

**Correct target:**
- **Upstream repo:** `triton-inference-server/core`
- **Target branch:** `main` (verify this is the default branch)
- **Your fork:** `YOUR_USERNAME/core` (created in Item 3)
- **Your branch:** `fix-8604-server-readiness`

**Verification checklist:**
- [ ] Forked `triton-inference-server/core` (not `server`)
- [ ] Applied patch to `src/server.cc` in the `core` repo
- [ ] Committed changes to your fork's branch
- [ ] When opening PR, base repo is `triton-inference-server/core`

---

## 🚀 Final Pre-Submission Checklist

Before opening the PR, verify:

- [ ] Items 1-2: Test hygiene fixes committed ✅
- [ ] Item 3: Patch compiles cleanly in `core` repo
- [ ] Item 4: Existing `L0_backend_python` tests pass
- [ ] Item 5: New test passes and reproduces the fix
- [ ] Item 6: CLA signed, PR targets correct repo
- [ ] PR description cleaned of "Track A/B" references (see Section 8 of validation report)
- [ ] Commit message follows upstream style (see Section 8 of validation report)

---

## ⚠️ Important Notes

1. **Test Location:** The new test may need to go in the `server` repo under `qa/`, not in the `core` repo. The `core` repo has minimal tests. Check if other `L0_*` tests exist in `core` or only in `server`.

2. **Two-Repo Strategy:** You may need TWO PRs:
   - PR #1 to `core`: The `src/server.cc` patch only
   - PR #2 to `server`: The test suite in `qa/L0_server_readiness/`
   
   Check how other cross-repo changes were handled (e.g., look at recent PRs that touch both repos).

3. **Build Time:** A full Triton build can take 30-60 minutes. Plan accordingly.

4. **Docker Shortcut:** If you have access to a pre-built Triton container, you can mount your patched `core` directory and skip the full rebuild. See Triton docs for container usage.

---

## 📞 Next Steps After Completion

Once Items 3-6 are done:

1. Push your changes to your `core` fork
2. Open the PR on `triton-inference-server/core`
3. Reference the validation report in the PR description (link to this repo's docs if public)
4. Respond promptly to any maintainer feedback
5. Be prepared to explain the type verification (`ModelIdentifier::name_`) if questioned

**Good luck! The core patch logic is sound — execution is now the only barrier.**
