# Inference branch

Image for the inference rig (2x AMD gfx1030/gfx1100 + 1x NVIDIA sm_120, 64G RAM).
The branch is recreated from upstream master whenever the fork is synced with upstream.

## Current state

- master base: cb7934c52 (2026-10-03)
- image: llama-cpp-cuda-rocm:inference-next9 (rollback: :inference-next8)
- Dockerfile: .devops/cuda-rocm10.Dockerfile (ROCm 10 from stable.repo.amd.com, CUDA 12.8.1)
- models.ini presets: Q8_0 (spec-type=draft-mtp, built-in head), Flash-Next Q3_K_XL
  (spec-type=draft-mtp with the ggml-org draft mtp-Qwen3.8-Flash-Next-Q4_0.gguf,
  lazy-mode=on), UD-Q6_K_XL (draft-mtp n-max=2)

## Cherry-picked PRs (all open upstream as of 2026-10-03)

| PR | what | author | commits here | why |
|---|---|---|---|---|
| #25592 | server: fix checkpoint handling for hybrid/recurrent models | krim | 1bf331632, 9647936cf | correct checkpoint snapshots/eviction for Qwen3.8 class |
| #26004 | server: preserve context checkpoints across slot save/restore | Amine pc2 | 98b1f8922 | checkpoints survive model LRU swaps |
| #29030 | qwen4exp/gemma4: gather lazy tensor rows with direct reads | Piotr Wilkin | c46e0208c (squash, 21 files) | replaces the mmap-backed lazy load of per_layer_tok_embd with per-context readers (llama-lazy-reader) that gather rows per ubatch; +65..121% pp in the PR's iGPU test. Conflict resolution vs master: kept master class names (llm_graph_input_qwen4exp_ple), took the PR's lazy_rows semantics; gemma4 keeps master's [TAG_GEMMA4_IMG_PADDING] comment; llama-model.cpp keeps master's head-block buft_list logic + PR's add_lazy_reader |

Devops commits (6c07035f6..42c930713): cuda-rocm10.Dockerfile + legacy-builder COPY fix.

## Excluded PRs

| PR | reason |
|---|---|
| #28243 | replaced: upstream merged #29761 (Qwen4Exp: add MTP, different implementation by Aman Gupta + ggerganov). The merged MTP needs the new ggml-org draft file mtp-Qwen3.8-Flash-Next-Q4_0.gguf, the old unsloth mtp-Q8_0.gguf is incompatible |
| #28213 | replaced: upstream rewrote the QSA path (#29751, build_attn_qsa now takes a sel mask + n_sel instead of top_k/qsa_bias/gather). Large-context Q8_0 behavior re-validated by the 220K battery after the next7 deploy |
| #28770 | merged upstream, now in master |
| #29805 | merged upstream (81e39ad34, 2026-10-01): clamp kpool re-pool bound, fixes the abort on the first decode of a cache-filling batch |
| #29825 | merged upstream (889edf43d, 2026-10-03): halve the indexer score memory + lightning indexer |
| #26592 | CUB path (hipCUB sum/mean/topk): caused the Flash HSA fault on ROCm 10 (2026-09-18) |
| #28330 | V-cache indexer skip: faulted on long prompts (2026-09-06) |
| #28136 | replaced by #29030 (in the branch) |
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
