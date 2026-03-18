# FlashMLA Cluster Reset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Recover the H100 FlashMLA validation workflow by recloning FlashMLA cleanly on the cluster, applying the minimal patch set, and rerunning the build/smoke job on the `research` partition.

**Architecture:** Keep our project worktree and `indoor` env unchanged. Treat the cluster `FlashMLA` checkout as disposable third-party state: recreate it cleanly, patch only the proven setup.py defects, then validate on H100 with a Slurm smoke job.

**Tech Stack:** Git, SSH, Conda, pip, Slurm, H100, FlashMLA

---

### Task 1: Cleanly replace the dirty cluster FlashMLA checkout

**Files:**
- Modify: cluster path `/home/amkrtchyan/src/FlashMLA`

**Step 1: Verify current checkout state**

Run:
```bash
ssh -t amkrtchyan@cluster.ysu.am 'bash -lc '"'"'cd /home/amkrtchyan/src/FlashMLA && git status --short && ls setup.py*'"'"''
```

Expected: local modifications are present and `setup.py.bak` may exist.

**Step 2: Replace the checkout with a fresh clone**

Run:
```bash
ssh -t amkrtchyan@cluster.ysu.am 'bash -lc '"'"'
  cd /home/amkrtchyan/src &&
  mv FlashMLA FlashMLA.dirty.$(date +%s) &&
  git clone https://github.com/deepseek-ai/FlashMLA.git FlashMLA
'"'"''
```

Expected: `/home/amkrtchyan/src/FlashMLA` is a clean upstream clone.

**Step 3: Verify clean baseline**

Run:
```bash
ssh -t amkrtchyan@cluster.ysu.am 'bash -lc '"'"'cd /home/amkrtchyan/src/FlashMLA && git status --short && git log --oneline -n 1'"'"''
```

Expected: no local changes.

**Step 4: Commit**

No git commit in our repo for this cluster-only task.

### Task 2: Apply the minimal FlashMLA setup.py patch cleanly

**Files:**
- Modify: cluster path `/home/amkrtchyan/src/FlashMLA/setup.py`

**Step 1: Write the patch script**

Create a remote helper script that:
- prefers `os.environ["CUDA_HOME"] or CUDA_HOME`
- promotes `DISABLE_SM100` / `DISABLE_SM90` to module scope
- filters `csrc/sm100/...` and `csrc/sm90/...` from the source list when disabled

**Step 2: Apply the patch**

Run the remote helper and print the patched sections:
```bash
ssh -t amkrtchyan@cluster.ysu.am 'bash -lc '"'"'python /home/amkrtchyan/fix_flashmla_setup.py && sed -n "1,120p" /home/amkrtchyan/src/FlashMLA/setup.py'"'"''
```

Expected: the patched file shows the three intended changes and nothing else.

**Step 3: Run a syntax sanity check**

Run:
```bash
ssh -t amkrtchyan@cluster.ysu.am 'bash -lc '"'"'python -m py_compile /home/amkrtchyan/src/FlashMLA/setup.py'"'"''
```

Expected: pass.

**Step 4: Commit**

No git commit in our repo for this cluster-only task.

### Task 3: Submit the H100 build-and-smoke job on `research`

**Files:**
- Create: cluster path `/home/amkrtchyan/flashmla_build_smoke.sh`

**Step 1: Write the Slurm script**

The script must:
- source conda
- `export NVCC_PREPEND_FLAGS=""`
- `conda activate indoor`
- `export PYTHONNOUSERSITE=1`
- `export CUDA_HOME="$CONDA_PREFIX"`
- `export FLASH_MLA_DISABLE_SM100=1`
- build FlashMLA with `python -m pip install -v --no-build-isolation -e .`
- run a minimal `flash_mla_sparse_fwd(...)` smoke

**Step 2: Submit the job**

Run:
```bash
ssh -t amkrtchyan@cluster.ysu.am 'bash -lc '"'"'sbatch --parsable /home/amkrtchyan/flashmla_build_smoke.sh'"'"''
```

Expected: returns a Slurm job id.

**Step 3: Poll the job**

Run:
```bash
ssh -t amkrtchyan@cluster.ysu.am 'bash -lc '"'"'squeue -j <JOBID> -o "%.18i %.9P %.8j %.8T %.10M %.6D %R" && tail -n 120 /home/amkrtchyan/slurm-flashmla-build-smoke-<JOBID>.out'"'"''
```

Expected: build finishes and smoke prints output shapes.

**Step 4: Commit**

No git commit in our repo for this cluster-only task.

### Task 4: Record findings back in our repo

**Files:**
- Modify: `docs/dsa_memory.md`

**Step 1: Update memory with the cluster result**

Add:
- `indoor` env details
- the FlashMLA setup.py upstream bug
- whether the clean re-clone + patch + research build succeeded

**Step 2: Run repo docs sanity check**

Run:
```bash
python -m py_compile src/networks/dsa_2d.py
```

Expected: pass; this confirms no local project code was changed in this slice.

**Step 3: Commit**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add docs/plans/2026-03-18-flashmla-cluster-reset-design.md docs/plans/2026-03-18-flashmla-cluster-reset-implementation.md docs/dsa_memory.md
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "docs(dsa): record FlashMLA cluster reset and validation results"
```
