# Inference branch

Image for the inference rig (2x AMD gfx1030/gfx1100 + 1x NVIDIA sm_120, 64G RAM).
The branch is recreated from upstream master whenever the fork is synced with upstream.

## Current state

- master base: 32bdc3a89 (2026-11-14, +16)
- prod image: llama-cpp-cuda-rocm:inference-next34 (ROCm 7.2.4, built from this branch with MARCH=znver3 + HIP_ARGSORT_CUB=1; rollback :inference-next33)
- NOTE 11-14: this branch adds the hipCUB wide-rows argsort (from the argsort-hip branch,
  commit b45e700af, built with HIP_ARGSORT_CUB=1). Wide argsort (e.g. the top-p backend
  sampler over the full ~151936 vocab) fell back to a CPU sampler on ROCm because the CUB
  path was excluded for HIP; this enables hipCUB for the argsort path only. On this rig's
  rocPRIM 4.2.0 (< 4.4.0) GGML_HIPCUB_CAPTURE_SAFE=0, so the capture guard keeps wide
  MULTI-row sorts (ne[0]>1024 && ne[1]>1) out of graph capture - the top-p sampler graph runs
  uncaptured (eager) while the model graphs keep CUDA graph capture. Gated by the
  HIP_ARGSORT_CUB build-arg (default 0/off), so a build without it is unchanged. No memory or
  quality regression (verified: same VRAM, needle 3/3, completion test correct).
- NOTE 11-14: the new master (+16) includes the DFlash speculative-decoding refactor
  (#32016, +802 lines in common/speculative.cpp) and #30021 (GCN MMQ configs, a different
  file from #31751). It also merged #31294 (spec p_min warning, squashed as 7874988b4) - so
  #31294 is dropped from this branch (it is now in master). #31746 stays dropped (its 91/62
  -line fattn-mma-f16.cuh rework conflicts with the master's MMA rework; re-add when the
  author rebases). #31234 stays on the OLD head (7617d05963b, quant-KV only): the new head
  (69b3edd40) generalized the tile load helpers into one templated function the kvrows fix
  does not match, and the rig uses Q8_0 KV (the quant path, in both) so the new head's
  bf16+tests are not needed here.
- NOTE 11-14: this branch adds the PP re-reservation fix (from the updated kvrows-wip,
  commit 73733559f): a re-reservation (e.g. DFlash feature setup after init) grows the
  worst-case graph, but the grown allocation is tested while the previous compute buffers
  are still held, so it can fail even when the card has enough total space. The fix releases
  the old buffers and retries pipeline parallelism once before falling back (src/llama-
  context.cpp, +11 lines). This matters on the rig: with kv-unified=1 the rows-FA workspace
  grows the graph, so the re-reservation can otherwise fail and force a non-PP fallback
  (~25-30% slower prefill).
- NOTE 11-11: the 11-09 "kv-unified=0 (temp)" workaround is FIXED and kv-unified is back to 1.
  Root cause of the intermittent ~3.5-5x prefill drops (180-220 t/s vs ~900-1400 normal,
  scaling with the other slot's unified-pool occupancy) was two defects in the tile-kernel
  rows port (fattn-tile.cuh), not the host code (upstream #29510's host side is correct):
  (1) an upper-bounded gather clamp `min(max(index,0), col0+i_sup-1)` that silently capped a
  gathered K/V row at the tile's last column (a no-op for contiguous single-seq rows, wrong
  for the 2-seq interleaved pool); (2) the KV-column tile offset double-added to the K/V load
  base when the gather (indices) already yields the global pool cell. Fixed to the MMA
  reference pattern: clamp only the -1 padding (`max(index,0)`) and drop the KV-column tile
  offset from the K/V base in rows mode (f16 + q8_0 quant paths, 3+4 sites). Also skip the
  mask scan for the kv_rows path (fattn-common.cuh: the scan reads a column tile past the
  end of the rows-sized mask). The kernel-selection gate (fattn.cu) routes the rows variant to
  the tile kernel on non-WMMA AMD (gfx1030 has no MMA). Verified locally on gfx1035 (tile port
  vs the upstream CPU reference, incl. 2-seq interleave + nkv=256 combine + Q8_0) and on the
  rig: needle 3/3, prefill over a full pool ~5-5.5x faster than the dense fallback, parallel
  2-slot decode unchanged. The kvrows-wip branch carries the fix + the PP re-reservation fix.
- Dockerfile: .devops/cuda-rocm.Dockerfile (ROCm 7.2.4; the 10.1 one is cuda-rocm10.Dockerfile)
- models.ini presets: Q8_0 (spec-type=draft-mtp, built-in head), Flash-Next Q3_K_XL
  (spec-type=draft-mtp with the ggml-org draft mtp-Qwen3.8-Flash-Next-Q4_0.gguf,
  lazy-mode=on), UD-Q6_K_XL (draft-mtp n-max=2)

## Cherry-picked PRs (all open upstream as of 2026-11-14)

Plus the kvrows tile-rows fix + the PP re-reservation fix (this branch's own work, not upstream
PRs), see the NOTEs above. See the `kvrows-wip` branch for the fix + its debug history

| PR | what | author | commits here | why |
|---|---|---|---|---|
| #31234 (old head 7617d05963b) | CUDA/HIP: support native quantized KV in flash_attn_tile | clarkbox | 2 commits (generalize dequantize_V + native quantized KV) | full K/V quant type matrix read natively in the tile FA kernel on CUDA and HIP. The new head (69b3edd40, +bf16 KV + tests) is dropped - see the 11-14 NOTE (it conflicts with the kvrows fix's tile load path). On the rig: Q8_0 cold pp +2.3% |
| #31751 | hip: avoid RDNA3 MMQ spills in the J=32 kernel | ravel7524 | 1 commit (DRAFT) | the Q8_0-layout MMQ J=32 kernels fully unroll the inner K loop and spill; keep it rolled (#pragma unroll 1). mmq-vec-dot.cuh (a different file from master's #30021 GCN MMQ configs, no conflict) |
| #31761 | hip: increase KV batch size for D256 tile attention | ravel7524 | 1 commit (DRAFT) | nbatch_K 64->128 for the two-column D256 tile on RDNA + occupancy hint 8->6. fattn-tile.cuh (config table, separate from the kvrows fix) |
| #25592 | server: checkpoint handling for hybrid/recurrent + drop unsafe rs rollback | krim | 2 commits (rebased to master, picked cleanly) | the pos_stale_min exact-erase for hybrid/recurrent checkpoints + drop the unsafe rollback fast-path. Re-added 2026-11-13 after the author rebased it |

Devops: the rig Dockerfiles (ROCm 7.2.4 + MARCH build-arg + BuildKit cache mount), restored on this base.

## Excluded PRs

| PR | reason |
|---|---|
| #31294 | merged upstream 2026-11-13 (squashed as 7874988b4, in master 32bdc3a89): spec p_min warning. Dropped from the branch - it is now in master. Our presets run p_min=0.00, so it was a functional no-op for us anyway |
| #31746 | DROPPED (conflict): a 91/62-line rework of fattn-mma-f16.cuh (D256 RDNA3 MMA tune) that conflicts with the master's MMA rework (F8_E4M3 + 40/72 head dims). Re-add once the author rebases it |
| #28243 | replaced: upstream merged #29761 (Qwen4Exp: add MTP). The merged MTP needs the new ggml-org draft file mtp-Qwen3.8-Flash-Next-Q4_0.gguf, the old unsloth mtp-Q8_0.gguf is incompatible |
| #28213 | replaced: upstream rewrote the QSA path (#29751). Large-context Q8_0 behavior re-validated by the 220K battery |
| #28770, #29805, #29825, #29901, #26004, #30087, #29353, #30804, #31506 | merged upstream, now in master |
| #30792 | replaced by #31234 (same feature, native quantized KV in the tile FA kernel) |
| #31669 | DROPPED 2026-11-06 (DRAFT/WIP, 7 conflicts in llama-graph/delta-net-base against master's new graph API). Modest value here (DFlash2 1-seq ~flat; multi-slot verify +6-34%). Re-add once the author updates the PR |
| #29963 (all three versions) | REJECTED three times on this rig (Q8_0 PP -35..-45%, Flash -41..-55%): the pipeline parallelism never enables for the Flash preset on the 3x mixed CUDA+ROCm rig, so only the downside lands. Do not revisit until the author can explain the non-NVIDIA regression |
| #29030 | closed upstream 2026-10-06 without merge, "superseded by #29599" (the mmap-backed lazy load already in master). Dropped per the owner's call |
| #26592 | CUB path (hipCUB sum/mean/topk): caused the Flash HSA fault on ROCm 10 (2026-09-18) |
| #28330 | V-cache indexer skip: faulted on long prompts (2026-09-06) |
| #28136 | replaced by master's native lazy load (#29599) |
| #27933 | closed upstream without merge (missing AI disclosure) |
| #27466 | in master |

## Recreate procedure (when the fork is synced)

1. `git fetch origin`, `git checkout master && git merge --ff-only origin/master`
2. PR refs: `git fetch https://github.com/ggml-org/llama.cpp pull/N/head:pr-N --force`
   for every PR in the table; also `git fetch origin kvrows-wip` (the fix + PP re-reservation
   live there)
3. check each PR upstream with `gh api repos/ggml-org/llama.cpp/pulls/N`:
   merged -> drop from the branch (it is in master); force-pushed -> take the new version
4. `git checkout -B inference master`, cherry-pick in the table order, then the kvrows fix +
   PP re-reservation commits (from the kvrows-wip branch), then the devops Dockerfiles
5. watch the new master delta for replacements: if a PR's subsystem was rewritten upstream
   (like the QSA/MTP/lazy-reader cases, or the FA-kernel rework vs #31234/#31746), drop the
   PR instead of force-merging conflicting signatures
6. update this file, commit
7. build on the rig (source tarball + `docker buildx build -f .devops/cuda-rocm.Dockerfile
   --build-arg MARCH=znver3 --build-arg HTTP_PROXY=http://172.17.0.1:58080`); a Dockerfile
   edit invalidates the compile cache from `COPY . .` onward
8. deploy: set the image tag in ~/projects/llamacpp/docker-compose.yml,
   `docker compose down && docker compose up -d --force-recreate`, wait /health=200
9. validate: `--list-devices` (3 GPUs), the needle test (3/3 codewords), the 2-phase
   parallel-decode bench (prefill over a full pool should be ~800-1200 t/s, not 220),
   model loaded + 0 HSA faults
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
