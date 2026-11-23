# Inference branch

Image for the inference rig (2x AMD gfx1030/gfx1100 + 1x NVIDIA sm_120, 64G RAM).
The branch is recreated from upstream master whenever the fork is synced with upstream.

## Current state

- master base: a0677c341 (2026-11-23, +5)
- prod image: llama-cpp-cuda-rocm:inference-next37 (ROCm 10.2, built from this branch with
  cuda-rocm10.Dockerfile, ROCM_CORE_VERSION=10.2 + MARCH=znver3 + HIP_ARGSORT_CUB=1;
  rollback :inference-next36)
- NOTE 11-23 (second sync): #31669 (GDN replay) added - it replaces the snapshot rollback with
  replay of the kept tokens for the delta-net (GDN) layers. NO CONFIG NEEDED: the gate is
  `n_rs_seq > 0 && arch in {qwen35, qwen35moe, qwen4exp, clover-moe, KDA models}` - all three of
  our models (Qwen3.8-27B, Qwen3.8-Flash-Next, the MTP draft) are arch `qwen35`, so replay is on
  automatically with speculative decoding; the old LLAMA_RS_REPLAY env flag was removed, there is
  no new ini/CLI knob. Verify with the log line `rs_replay = true` at model init. Expected on the
  rig: the recurrent-state memory per sequence drops (PR: -73% at n_rs_seq 3, -85% at 7, per-seq
  state ~200 MiB -> ~50 MiB class) and the 2-slot DFlash2 decode is flat-to-slightly-faster
  (PR server numbers: +2.1% at 4 slots, +7.9% at 16). qwen3next models stay on snapshots.
- NOTE 11-23 (second sync): master +5 touched none of our files; all picks clean. #31669's head
  was 2 commits behind the master tip (ui PWA naming) - cherry-picked the PR's 16 commits
  (51fc2f90e..fe7a725fb) as one commit, NOT a tree diff (a raw diff would have reverted the PWA
  commit). Rig proxy still 503-ing - built direct.
- NOTE 11-23: ROCm 10.2 is the production base now (user soak-decided after :inference-next35-rocm102
  ran clean for days). Why: ROCm 10.2.0 bundles rocPRIM 4.8.0 >= 4.4.0, so the build gets
  GGML_HIPCUB_CAPTURE_SAFE=1 and the wide top-p argsort no longer knocks the sampler graph out of
  capture (7.2.4/rocPRIM 4.2.0 ran it eager). Measured vs 7.2.4: full-pool prefill 1228.7 vs
  1162.7 t/s, parallel 2-slot decode 19.57 vs 16.75 t/s (+17%), needle 3/3, 0 HSA faults.
- NOTE 11-23: clean round - master's +279 touched NONE of the fattn/argsort/hip/CMake files
  (the kvrows fix + #31234-old + argsort replayed byte-identical, zero conflicts), only
  llama-context.cpp (5 MoE commits, PP fix applied clean) and the UI (Llama Voice sources,
  #32680 - the rig Dockerfiles now stage ui/lib/voice/src next to ui/lib/ui for the npm build).
  All 5 tracked PR heads unchanged upstream - no re-picks needed.
- NOTE 11-19: the new master (+171) includes the K/V read-order rotation for L2 reuse
  (#32407, merged 11-17, fattn-mma-f16.cuh) - NVIDIA Ampere+ only (rotate = nstages > 1,
  cp_async is NVIDIA-only), so it speeds up long-context prefill on the 5070 Ti (1.07-1.41x
  on the 5090's FA at 32K-128K) and does nothing on the AMD cards. It came in for free.
- NOTE 11-19: master also touched the fattn files this round (#30672 SKU-aware FP16/BF16
  accumulation gating, #31617 warp-primitives refactor, #32217 Inkling 2 with the
  mask->ne[2] == 1 scan/gqa guards). The #31234-old + kvrows-fix cherry-picks had conflicts
  there, resolved by keeping BOTH the upstream mask->ne[2] == 1 guards and the #31234
  ncols1_eff == ncols1 / pad_gqa additions (the scan-skip commit then combines all three
  guards with !kv_rows). #25592 (rebased by the author) conflicted with master's own
  n_swa_reuse rename in server-context.cpp - resolved by passing both n_swa_reuse and the
  new ctx_tgt_state_exact() to get_reuse (they're complementary: n_swa_reuse = n_swa=0 for
  hybrid/recurrent, ctx_tgt_state_exact = checkpoint-pos semantics).
- NOTE 11-19: #31234 got force-pushed again (e40ab5c81, q4_0+q8_0 KV in one unified
  templated tile load helper) - still conflicts with the kvrows fix, so the branch keeps
  the OLD head (quant-KV, 2 commits). Re-apply the kvrows gather to the new unified helper
  before adopting the new head. #31746 force-pushed (merged fresh master into its branch)
  but still conflicts with the post-#32407 fattn-mma-f16.cuh - stays DROPPED.
- NOTE 11-14: the new master also included the DFlash speculative-decoding refactor (#32016)
  and merged #31294 (7874988b4, dropped from the branch).
- NOTE 11-14: the PP re-reservation fix (73733559f) is in this branch: a re-reservation
  (e.g. DFlash feature setup after init) grows the worst-case graph, but the grown
  allocation is tested while the previous compute buffers are still held, so it can fail
  even when the card has enough total space. The fix releases the old buffers and retries
  pipeline parallelism once before falling back (src/llama-context.cpp, +11 lines).
- NOTE 11-14: hipCUB wide-rows argsort (b45e700af, built with HIP_ARGSORT_CUB=1): the top-p
  backend sampler over the ~151936 vocab runs on the GPU instead of the CPU sampler. On
  rocPRIM < 4.4.0 (the rig's 7.2.4 ships 4.2.0) the capture guard keeps wide multi-row
  sorts (ne[0]>1024 && ne[1]>1) out of graph capture. ON ROCm 10.2: check the shipped
  rocPRIM version - if >= 4.4.0 then GGML_HIPCUB_CAPTURE_SAFE=1 and the sampler graph is
  fully captured (native argsort, no eager fallback). cuda-rocm10.Dockerfile now takes
  --build-arg ROCM_CORE_VERSION (default 10.1) and the HIP_ARGSORT_CUB arg.
- NOTE 11-11: the kvrows tile-rows fix (3 commits) is in this branch: it makes the unified
  KV pool (kv-unified=1) run the fast rows FA on RDNA2 (tile kernel, native Q8_0) and RDNA3
  (MMA F16) instead of the dense fallback over the whole pool (the ~3.5-5x prefill drops).
  Two original defects (upper-bounded gather clamp + double-added KV-column tile offset)
  were fixed to the MMA reference pattern; the mask scan is skipped for the rows path.
  Verified on the rig: needle 3/3, prefill over a full pool ~5.5x faster than the dense
  fallback, parallel 2-slot decode unchanged.
- Dockerfiles: .devops/cuda-rocm.Dockerfile (ROCm 7.2.4), .devops/cuda-rocm10.Dockerfile
  (ROCm 10.x via --build-arg ROCM_CORE_VERSION, AMD stable.repo.amd.com packages)
- models.ini presets: Q8_0 (spec-type=draft-mtp, built-in head), Flash-Next Q3_K_XL
  (spec-type=draft-mtp with the ggml-org draft mtp-Qwen3.8-Flash-Next-Q4_0.gguf,
  lazy-mode=on), UD-Q6_K_XL (draft-mtp n-max=2)

## Cherry-picked PRs (all open upstream as of 2026-11-23)

Plus the kvrows tile-rows fix + the PP re-reservation fix + the hipCUB argsort (this
branch's own work, not upstream PRs). See the `kvrows-wip` and `argsort-hip` branches for
the fix + its debug history.

| PR | what | author | commits here | why |
|---|---|---|---|---|
| #31234 (old head 7617d05963b) | CUDA/HIP: support native quantized KV in flash_attn_tile | clarkbox | 2 commits (generalize dequantize_V + native quantized KV) | full K/V quant type matrix read natively in the tile FA kernel. The new head (e40ab5c81) is dropped - see the 11-19 NOTE (unified load helper conflicts with the kvrows fix). On the rig: Q8_0 cold pp +2.3% |
| #31751 | hip: avoid RDNA3 MMQ spills in the J=32 kernel | ravel7524 | 1 commit (DRAFT) | the Q8_0-layout MMQ J=32 kernels fully unroll the inner K loop and spill; keep it rolled (#pragma unroll 1). mmq-vec-dot.cuh |
| #31761 | hip: increase KV batch size for D256 tile attention | ravel7524 | 1 commit (DRAFT) | nbatch_K 64->128 for the two-column D256 tile on RDNA + occupancy hint 8->6. fattn-tile.cuh (config table) |
| #25592 (rebased, 4ac1aeab1) | server: checkpoint handling for hybrid/recurrent + drop unsafe rs rollback | krim | 2 commits | pos_min = pos_max for FULL/RS checkpoints + the ctx_tgt_state_exact() get_reuse semantics + drop the unsafe rollback fast-path. Re-picked 2026-11-19 after the author rebased (conflict with master's n_swa_reuse, resolved) |
| #31669 (fe7a725fb, approved, near-merge) | GDN: use replay instead of rollback | amangupta052 | 16 commits, squashed to 1 | the delta-net (GDN) layers roll back by replaying the kept tokens instead of keeping n_rs_seq snapshots: -73..-91% recurrent-state memory per sequence, flat-to-+8% decode. All our models are arch qwen35 => replay on with no config. Cherry-picked 2026-11-23 (the Nov-6 WIP version with 7 conflicts is long superseded) |

Devops: the rig Dockerfiles (ROCm 7.2.4 + ROCm 10.x version arg + MARCH build-arg + BuildKit cache mount + HIP_ARGSORT_CUB arg), restored/kept on this base.

## Excluded PRs

| PR | reason |
|---|---|
| #32407 | MERGED upstream 2026-11-17 (054a1babf, in master f8caa5ea3): K/V read-order rotation for L2 reuse in the FP16 MMA FA. NVIDIA Ampere+ only (rotate needs nstages>1, cp_async is NVIDIA-only) - the 5070 Ti gets it, the AMD cards don't |
| #31294 | merged upstream 2026-11-13 (7874988b4, in master): spec p_min warning. Dropped - it is in master. Our presets run p_min=0.00, a no-op for us |
| #31746 | DROPPED (conflict, still after the author's 11-16 rebase-merge): a 91/62-line rework of fattn-mma-f16.cuh (D256 RDNA3 MMA tune) that conflicts with the post-#32407 master MMA kernel. Re-add once the author rebases onto the #32407-included master |
| #28243 | replaced: upstream merged #29761 (Qwen4Exp: add MTP). The merged MTP needs the new ggml-org draft file mtp-Qwen3.8-Flash-Next-Q4_0.gguf |
| #28213 | replaced: upstream rewrote the QSA path (#29751). Large-context Q8_0 behavior re-validated by the 220K battery |
| #28770, #29805, #29825, #29901, #26004, #30087, #29353, #30804, #31506 | merged upstream, now in master |
| #30792 | replaced by #31234 (same feature, native quantized KV in the tile FA kernel) |
| #29963 (all three versions) | REJECTED three times on this rig (Q8_0 PP -35..-45%, Flash -41..-55%): the pipeline parallelism never enables for the Flash preset on the 3x mixed CUDA+ROCm rig. Do not revisit until the author can explain the non-NVIDIA regression |
| #29030 | closed upstream 2026-10-06 without merge, "superseded by #29599" (the mmap-backed lazy load already in master) |
| #26592 | CUB path (hipCUB sum/mean/topk): caused the Flash HSA fault on ROCm 10 (2026-09-18) |
| #28330 | V-cache indexer skip: faulted on long prompts (2026-09-06) |
| #28136 | replaced by master's native lazy load (#29599) |
| #27933 | closed upstream without merge (missing AI disclosure) |
| #27466 | in master |

## Recreate procedure (when the fork is synced)

1. `git fetch origin`, `git checkout master && git merge --ff-only origin/master`
2. PR refs: `git fetch https://github.com/ggml-org/llama.cpp pull/N/head:pr-N --force`
   for every PR in the table; also `git fetch origin kvrows-wip argsort-hip`
3. check each PR upstream with `gh api repos/ggml-org/llama.cpp/pulls/N`:
   merged -> drop from the branch (it is in master); force-pushed -> examine the new head
   (if its subsystem changed, e.g. #31234's unified load helper, keep the old head)
4. `git checkout -f -B inference master`, cherry-pick in the table order, then the kvrows
   fix (3) + PP re-reservation (1) + argsort (1) commits from the kvrows-wip/argsort-hip
   branches, then restore the devops Dockerfiles if the argsort commit's Dockerfile hunk
   conflicted with the upstream one (`git checkout <old-inference> -- .devops/cuda-rocm*.Dockerfile`)
5. watch the new master delta for fattn/server file changes: since 11-19 the master keeps
   evolving fattn-common/fattn-tile (Inkling 2 mask guards) and the server checkpoint code
   (n_swa_reuse), which conflict with #31234-old + the kvrows scan-skip + #25592
6. update this file, commit
7. build on the rig (source tarball + `docker buildx build -f .devops/cuda-rocm.Dockerfile
   --build-arg MARCH=znver3 --build-arg HIP_ARGSORT_CUB=1 --build-arg HTTP_PROXY=http://172.17.0.1:58080`);
   the ROCm 10.x variant: `-f .devops/cuda-rocm10.Dockerfile --build-arg ROCM_CORE_VERSION=10.2`
8. deploy: set the image tag in ~/projects/llamacpp/docker-compose.yml,
   `docker compose down && docker compose up -d --force-recreate`, wait /health=200
9. validate: `--list-devices` (3 GPUs), the needle test (3/3), the 2-phase
   parallel-decode bench (prefill over a full pool should be ~800-1200 t/s, not 220),
   the quality test (4/4), model loaded + 0 HSA faults + no non-PP fallback
10. push inference only after validation

## Rollback

- previous good image tag (see "Current state")
- compose backups: ~/projects/llamacpp/docker-compose.yml.bak-*
- after an HSA fault the child process dies and needs a restart (the parent re-spawns it)

## Gotchas

- the rig's .env LLAMACPP_API_KEY holds comma-separated keys; curl with one fragment
  (`cut -d, -f1`), not the whole string
- container healthcheck override: port 8000 (the image default probes 8080)
- compose group_add: ["video", "987"] (Arch render gid)
- models.ini presets are read by the parent server at startup: after editing the file,
  `docker compose restart llama.cpp` (the service name in compose is "llama.cpp")
- source tarball for the rig build: exclude top-level `./.git ./build* ./models ./.cache
  ./.cortexkit` only (a broad `--exclude=models` removes tools/ui's models module and breaks
  the TUI npm build)
- image tag in compose must be replaced with a sed that matches hyphens
  (`[a-zA-Z0-9_-]*`, NOT `[a-z0-9]+` - "rocm724" was mangled once and prod was left on an
  experimental image with garbage output); ALWAYS verify with
  `docker inspect llama-server --format {{.Config.Image}}` after every switch
