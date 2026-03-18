# FlashMLA Cluster Reset Design

**Goal:** Recover the H100 FlashMLA validation path on the cluster by discarding the dirty third-party checkout, recloning FlashMLA cleanly, applying a minimal local patch set, and rerunning the build/smoke flow on the `research` partition.

**Current State**
- The cluster repo worktree for our project is healthy at `/home/amkrtchyan/src/mlsp_wair_d/.worktrees/nsa-triton-longseq-investigation`.
- The user-owned cluster env `indoor` is the correct base env.
- `indoorolo` is stale and should be ignored.
- `FlashMLA` on the cluster is currently dirty from interrupted patching and should not be trusted.

**Root Causes**
- `FlashMLA/setup.py` has an upstream bug for our toolchain path: `FLASH_MLA_DISABLE_SM100=1` affects arch flags but does not remove `csrc/sm100/...` from the source list.
- Earlier cluster Python state was contaminated by `~/.local/lib/python3.12/site-packages`, which shadowed the conda env torch install.
- The login-node FlashMLA build died in heavy native compilation; this work belongs on the `research` partition.

**Approved Approach**
1. Do not continue patching the current dirty `FlashMLA` checkout.
2. Remove or sideline the current `/home/amkrtchyan/src/FlashMLA` checkout and reclone it fresh.
3. Reapply only the minimal required cluster-local patch set:
   - prefer `os.environ["CUDA_HOME"]` over the imported helper fallback when resolving `nvcc`
   - filter `csrc/sm100/...` and `csrc/sm90/...` sources based on `FLASH_MLA_DISABLE_SM100` / `FLASH_MLA_DISABLE_SM90`
4. Build and smoke-test FlashMLA only from the clean `indoor` env with `PYTHONNOUSERSITE=1`.
5. Run the build and sparse-prefill smoke on the `research` partition, not on the login node.

**Non-Goals**
- No changes to our project code in this slice.
- No backward integration yet.
- No selector-kernel integration yet.

**Success Criteria**
- `FlashMLA` installs in the `indoor` env on the cluster from a fresh checkout.
- The H100 `research` job imports `flash_mla` successfully.
- A minimal `flash_mla_sparse_fwd(...)` smoke runs on H100 and prints output tensor shapes.
- No further changes are made to the broken `indoorolo` path.
