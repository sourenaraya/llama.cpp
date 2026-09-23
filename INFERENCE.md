# Inference branch

Image for the inference rig (2x AMD gfx1030/gfx1100 + 1x NVIDIA sm_120, 64G RAM).
The branch is recreated from upstream master whenever the fork is synced with upstream.

## Current state

- master base: 86b2daa73 (2026-09-22)
- image: llama-cpp-cuda-rocm:inference-next5 (rollback: :mtpfix)
- Dockerfile: .devops/cuda-rocm10.Dockerfile (ROCm 10 from stable.repo.amd.com, CUDA 12.8.1)
- models.ini presets: Q8_0 (spec-type=draft-mtp, built-in head), Flash-Next Q3_K_XL
  (spec-type=none since 2026-09-22: measured +25% PP, no TG gain; MTP draft file
  mtp-*.gguf available but not wired), UD-Q6_K_XL (draft-mtp n-max=2)

## Cherry-picked PRs (all open upstream as of 2026-09-22)

| PR | what | author | commits here | why |
|---|---|---|---|---|
| #25592 | server: fix checkpoint handling for hybrid/recurrent models | krim | f6508a895, 3e9901e08 | correct checkpoint snapshots/eviction for Qwen3.8 class |
| #26004 | server: preserve context checkpoints across slot save/restore | Amine pc2 | 898fe4439 | checkpoints survive model LRU swaps |
| #28213 | qwen4exp: gather-based sparse attention for QSA decode | Abdel Darwish | 4b29618cf | required for Q8_0 large context on ROCm 10 |
| #29030 | gather lazy tensor rows with direct reads | Piotr Wilkin | 82bfef8ce | PLE direct reads (--lazy-mode on-direct), replaces #28136 |
| #28243 | models: Qwen3.8-Flash-Next MTP | danielhanchen | 429675394 (squash of net diff vs bb3c853c3) | MTP draft head for Flash-Next; includes the 09-21 review fixes (draft load path, nextn_layer_offset) |

Devops commits (3710cbcb2..fef303a82): cuda-rocm10.Dockerfile + legacy-builder COPY fix.

## Excluded PRs

| PR | reason |
|---|---|
| #28770 | merged upstream, now in master |
| #26592 | CUB path (hipCUB sum/mean/topk): caused the Flash HSA fault on ROCm 10 (2026-09-18) |
| #28330 | V-cache indexer skip: faulted on long prompts (2026-09-06) |
| #28136 | replaced by #29030 (same feature, different design) |
| #27933 | closed upstream without merge (missing AI disclosure) |
| #27466 | in master |

## Recreate procedure (when the fork is synced)

1. `git fetch origin`, `git checkout master && git merge --ff-only origin/master`
2. PR refs: `git fetch https://github.com/ggml-org/llama.cpp pull/N/head:pr-N --force` for every PR in the table
3. check each PR upstream: merged -> drop from the branch (it is in master);
   force-pushed -> take the new version (net diff vs merge-base, or fresh cherry-pick)
4. `git checkout -B inference master`, cherry-pick in the table order:
   #25592 (2 commits), #26004, #28213, the 5 devops commits, #29030, then #28243 as a
   squash net-diff (preserve the author)
5. sanity: `git show pr-28243:common/speculative.cpp` must equal the branch file (md5),
   qwen4exp.cpp diff vs the PR head must contain only the #28213 hunks
6. update this file, commit
7. build on the rig (source tarball + `docker build -f .devops/cuda-rocm10.Dockerfile
   --build-arg BUILD_JOBS=8`; the rig has the legacy docker builder, a Dockerfile edit
   invalidates the compile cache from COPY . . onward)
8. deploy: set the image tag in ~/projects/llamacpp/docker-compose.yml,
   `docker compose up -d --build --force-recreate`
9. validate: `--list-devices` (3 GPUs), Q8_0 completion test (expect 361), Flash short +
   long generation, MTP draft acceptance in the server log
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
