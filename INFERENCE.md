# Inference branch

Image for the inference rig (2x AMD gfx1030/gfx1100 + 1x NVIDIA sm_120, 64G RAM).
The branch is recreated from upstream master whenever the fork is synced with upstream.

## Current state

- master base: 79e2e74eb (2026-10-21)
- prod image: llama-cpp-cuda-rocm:inference-next15 (ROCm 10.1, built from this branch; rollback :inference-next14)
- Dockerfile: .devops/cuda-rocm10.Dockerfile (ROCm 10 from stable.repo.amd.com, CUDA 12.8.1)
- models.ini presets: Q8_0 (spec-type=draft-mtp, built-in head), Flash-Next Q3_K_XL
  (spec-type=draft-mtp with the ggml-org draft mtp-Qwen3.8-Flash-Next-Q4_0.gguf,
  lazy-mode=on), UD-Q6_K_XL (draft-mtp n-max=2)

## Cherry-picked PRs (all open upstream as of 2026-10-21)

| PR | what | author | commits here | why |
|---|---|---|---|---|
| #25592 | server: fix checkpoint handling for hybrid/recurrent models | krim | (2 commits, re-applied on this base) | correct checkpoint snapshots/eviction for Qwen3.8 class |
| #29353 | cuda, hip: GDN chunked kernel (MMA) | am17an | (4 commits, re-applied on this base) | new MMA-based chunked GDN prefill kernel, enabled on sm86/sm89/sm120/sm121 (our 5070 Ti = sm120 gets it; HIP keeps the AR path); +10..19% pp in the PR's Qwen3.8-27B/Qwen3.5-35B benches. mma.cuh ported over the #29612 swizzle refactor: the PR's 5-arg swz ldmatrix helpers kept verbatim with the runtime-stride swizzle_bytes/swizzle<bool> helpers re-added (self-contained, the kernel was validated against exactly that swizzle). The 2026-10-21 rebase reuses the same resolution; the #30087 merge (gdn_cols_per_warp=4) made the launcher conflict smaller |


Devops commits (a1fc9b2ed..3eb8d625b): cuda-rocm10.Dockerfile + legacy-builder COPY fix.

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
| #29030 | closed upstream 2026-10-06 without merge, "superseded by #29599" (the mmap-backed lazy load already in master). Dropped from the branch on 2026-10-07 per the owner's call: the Flash preset runs on master's native lazy mode (`on`), the direct-reads perf (+65..121% pp on iGPU testbeds) is lost on this rig until it comes back upstream in some form |
| #29963 | tested 2026-10-05, REJECTED on this rig: (1) pipeline parallelism never enabled for the Flash preset (no "pipeline parallelism enabled" log, gates fail silently - fit n_gpu_layers / host-weight spread on 3x mixed rig); (2) without the pipeline benefit it still cuts Q8_0 (qwen35 27B, fully in VRAM) PP8192 by ~35%: 1470 t/s on the 1537a0a8b branch -> 823-955 t/s with the PR (same 11341-token cold-prompt test; clean master d89651a7b = 1461, so the +23 master drift is innocent, the PR itself is the regression). Author's 3x NVIDIA PCIe5 rig shows x1.85-1.9 on long Flash prompts, no all-in-VRAM regression - the Q8_0 hit looks specific to the CUDA+ROCm layer-split rig. Revisit if the author fixes the non-pipeline regression and the Flash pipeline gates |
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
