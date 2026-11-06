# Inference branch

Image for the inference rig (2x AMD gfx1030/gfx1100 + 1x NVIDIA sm_120, 64G RAM).
The branch is recreated from upstream master whenever the fork is synced with upstream.

## Current state

- master base: 67107ac34 (2026-11-06, +48)
- prod image: llama-cpp-cuda-rocm:inference-next28 (ROCm 10.1, built from this branch with MARCH=znver3; rollback :inference-next27)
- Dockerfile: .devops/cuda-rocm10.Dockerfile (ROCm 10 from stable.repo.amd.com, CUDA 12.8.1)
- models.ini presets: Q8_0 (spec-type=draft-mtp, built-in head), Flash-Next Q3_K_XL
  (spec-type=draft-mtp with the ggml-org draft mtp-Qwen3.8-Flash-Next-Q4_0.gguf,
  lazy-mode=on), UD-Q6_K_XL (draft-mtp n-max=2)

## Cherry-picked PRs (all open upstream as of 2026-10-20)

| PR | what | author | commits here | why |
|---|---|---|---|---|
| #31234 | CUDA/HIP: support native quantized KV in flash_attn_tile | clarkbox | (2 commits, author's 2026-11-02 rewrite incl. the RDNA dp4a follow-up) | full K/V type matrix (f16/bf16/q4_0..q8_0) read natively in the tile FA kernel on CUDA and HIP, one instance file per type pair + generator script; decode and small batches use the tile kernel, large prefill batches keep the f16 conversion path. Author's R9700 numbers: tg x1.02-1.6 (q8_0 KV, context 8K-128K), pp unchanged. On the rig (next23): Q8_0 44.6K cold pp +2.3% vs pre-PR image |
| #31294 | spec: sampled-draft p_min | Pranesh Gonegandla | (2 commits, final version 2026-11-02: the p_min skip was replaced by a startup warning - p_min is applied as in master, the PR only warns to set it to 0 with probabilistic sampling) | our presets already run p_min=0.00, so both PR versions were functionally no-ops for us |
| #31746 | hip: tune D256 MMA FlashAttention on RDNA3 | (1 commit, DRAFT) | D256 64-col FA tiles on gfx1100/gfx1101: Q into shared memory, smaller K/V batches, WMMA barriers. ROCm 10.1 census: the dense <256,256,32,2> kernel used by Qwen3.8-27B went from 564 VGPR spills + 1140B scratch to zero. fattn-mma-f16.cuh +24 |
| #31751 | hip: avoid RDNA3 MMQ spills in the J=32 kernel | (1 commit, DRAFT) | the Q8_0-layout MMQ J=32 kernels fully unroll the inner K loop and spill; keep it rolled (#pragma unroll 1) with the same accumulation order. Author's W7800 numbers: PP17/PP32 +36%, 28-col MUL_MAT -41%. mmq-vec-dot.cuh |
| #31761 | hip: increase KV batch size for D256 tile attention | (1 commit, DRAFT) | nbatch_K 64->128 for the two-column D256 tile on RDNA + occupancy hint 8->6; halves the K staging passes. Author's W7800 numbers: decode at 131K +13.5%. fattn-tile.cuh 1 line |



Devops commits (7, re-applied on this base): cuda-rocm10.Dockerfile (ROCm 10.1, MARCH build-arg) + legacy-builder COPY fix.

## Excluded PRs

| PR | reason |
|---|---|
| #28243 | replaced: upstream merged #29761 (Qwen4Exp: add MTP, different implementation by Aman Gupta + ggerganov). The merged MTP needs the new ggml-org draft file mtp-Qwen3.8-Flash-Next-Q4_0.gguf, the old unsloth mtp-Q8_0.gguf is incompatible |
| #28213 | replaced: upstream rewrote the QSA path (#29751, build_attn_qsa now takes a sel mask + n_sel instead of top_k/qsa_bias/gather). Large-context Q8_0 behavior re-validated by the 220K battery after the next7 deploy |
| #28770 | merged upstream, now in master |
| #29805 | merged upstream (81e39ad34, 2026-10-01): clamp kpool re-pool bound, fixes the abort on the first decode of a cache-filling batch |
| #29825 | merged upstream (889edf43d, 2026-10-03): halve the indexer score memory + lightning indexer |
| #29901 | merged upstream (1b43d3116, in the 2026-10-07 sync): tiled lightning indexer for 4 heads |
| #26004 | merged upstream (033df86b6, 2026-10-08): checkpoints across slot save/restore |
| #30087 | merged upstream (24e41838e, 2026-10-08): GDN columns per warp. The merged version is simpler than the PR head we had cherry-picked: a single gdn_cols_per_warp=4 constant instead of 2 (4 at S_v=128) |
| #29353 | merged upstream (62df1815b, 2026-10-14): GDN chunked MMA kernel. The author rebased and updated it before merge (head dde6a4d93, ours was 377e111e8); the merged version is in master, our manual mma.cuh swizzle port is obsolete |
| #30804 | merged upstream (280141e6d, 2026-10-23): chunk graphs. The merged version carries the invalidate_cuda_graphs fix for master's #26771 that we had to patch in 2026-10-22 (bfea0596a) - the patch is obsolete |
| #30792 | replaced by #31234 (2026-10-27): same feature (native quantized KV in the tile FA kernel), #31234 is the CUDA+HIP generalization with the full K/V type matrix |
| #31506 | merged upstream (806ae1a47, 2026-11-03): DFlash/DSpark hidden features via device copies. We had cherry-picked it 2026-11-02 (next25); the merged version is now in master, the cherry-pick is obsolete |
| #25592 | DROPPED 2026-11-06 (per the owner: too much conflict). Master absorbed its machinery: common_prompt_checkpoint now carries id_task/data_spec/data_dft/load_tgt/load_dft (via #30137/#30141/#31350/#30666), and #30137 (2026-11-05) switched checkpoints to shared_ptr<const> (reference counting) - the PR's adopt pattern (writing id_task after creation) cannot compile against the const API. Remaining unique part: the pos_stale_min exact-erase for hybrid/recurrent - covered by master's do_reset fallback (full prompt reprocess) at the cost of a possible PP hit in rare hybrid cases. Re-check at the next sync if #25592 gets reworked |
| #31669 | DROPPED 2026-11-06 (DRAFT/WIP, 7 conflicts on the 2026-11-06 master base in llama-graph/delta-net-base - the replay mechanism collides with master's new graph API). We had cherry-picked it 2026-11-04 (next27, worked: 0 faults, 9477). Its value here is modest (DFlash2 1-seq ~flat; multi-slot verify +6-34%). Re-add once the author updates the PR against the new master |
| #29963 (all three versions) | REJECTED three times on this rig: 2026-10-05 (Q8_0 PP8192 1470 -> 823-955 t/s, -35%, Flash -31%), 2026-10-15 rewrite (Q8_0 -45%, Flash -55%), 2026-10-17 rewrite (Q8_0 680.5 vs next19 1042.5 t/s, -35%, Flash 219.7 vs ~370, -41%). In every version the pipeline parallelism never enables for the Flash preset (no "pipeline" log line, gates fail silently on the 3x mixed CUDA+ROCm rig), so only the downside lands: the sched changes (upload-inputs-once-per-device, host-weight offload callback) also cut PP on the dense all-in-VRAM Q8_0. The 10-17 version's new callback view-tensor fix changed nothing. Author's 3x NVIDIA PCIe5 rig shows the opposite (x1.85-1.9). Do not revisit until the author can explain the non-NVIDIA regression and the Flash pipeline gates |
| #29030 | closed upstream 2026-10-06 without merge, "superseded by #29599" (the mmap-backed lazy load already in master). Dropped from the branch on 2026-10-07 per the owner's call: the Flash preset runs on master's native lazy mode (`on`), the direct-reads perf (+65..121% pp on iGPU testbeds) is lost on this rig until it comes back upstream in some form |
| #26592 | CUB path (hipCUB sum/mean/topk): caused the Flash HSA fault on ROCm 10 (2026-09-18) |
| #28330 | V-cache indexer skip: faulted on long prompts (2026-09-06) |
| #28136 | replaced by master's native lazy load (#29599); the #29030 direct-reads extension was closed without merge |
| #27933 | closed upstream without merge (missing AI disclosure) |
| #27466 | in master |

## Recreate procedure (when the fork is synced)

1. `git fetch origin`, `git checkout master && git merge --ff-only origin/master`
2. PR refs: `git fetch https://github.com/ggml-org/llama.cpp pull/N/head:pr-N --force`
   for every PR in the table
3. check each PR upstream with `gh api repos/ggml-org/llama.cpp/pulls/N`:
   merged -> drop from the branch (it is in master); force-pushed -> take the new
   version (net diff vs merge-base, or fresh cherry-pick)
4. `git checkout -B inference master`, cherry-pick in the table order:
   #25592 (2 commits), #26004, the 5 devops commits.
   Multi-commit PRs that are not cherry-pickable (merge commits inside) go in as a
   squash net diff vs the merge-base, author preserved
5. watch the new master delta for replacements: if a PR's subsystem was rewritten
   upstream (like the QSA/MTP/lazy-reader cases above), drop the PR instead of
   force-merging conflicting signatures
6. update this file, commit
7. build on the rig (source tarball + `docker build -f .devops/cuda-rocm10.Dockerfile`;
   the rig has the legacy docker builder, a Dockerfile edit invalidates the compile
   cache from COPY . . onward)
8. deploy: set the image tag in ~/projects/llamacpp/docker-compose.yml,
   `docker compose up -d --build --force-recreate`
9. validate: `--list-devices` (3 GPUs), Q8_0 completion test (expect 361, use
   /no_think - note: as of 2026-09-28 master /no_think no longer disables thinking,
   max_tokens no longer counts reasoning tokens), Flash short + long generation,
   MTP draft acceptance in the server log, and the 220K battery for Q8_0 large context
10. push inference only after validation

## Rollback

- previous good image tag (see "Current state")
- compose backups: ~/projects/llamacpp/docker-compose.yml.bak-*
- after an HSA fault the child process dies and needs a restart (the parent re-spawns it)

## Gotchas

- the rig's .env LLAMACPP_API_KEY holds three comma-separated keys; curl with one fragment
  (`cut -d, -f1`), not the whole string
- container healthcheck override: port 8000 (the image default probes 8080)
- compose group_add: ["video", "987"] (Arch render gid)
- 64G RAM budget: dio weights ~28G + KV + prompt cache + MTP; keep ctx-checkpoints <= 8
  under MTP or a large request can OOM-kill the child
- models.ini presets are read by the parent server at startup: after editing the file,
  `docker compose restart llama.cpp` (the service name in compose is "llama.cpp")
