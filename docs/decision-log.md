# Decision log

What I chose, what I chose it over, and why. "Tested" means the alternative was measured with the eval suite and lost.

| Decision / Tradeoff | Why I chose it |
|---|---|
| Used **Claude Haiku 4.5** instead of a larger model | Cheaper and faster. About 1 cent per answer, so I could re-run the full eval after every change. The hard part was retrieval, not reasoning. |
| Used **local `all-MiniLM-L6-v2` embeddings** instead of an embeddings API | Free, offline, no second API key, small enough to bake into the image. Misses some paraphrases ("restart" vs "replacement Pods"). |
| Used **embedded Chroma** instead of a vector DB server | Docs change rarely and the app only reads the index, so a separate service adds nothing. New docs need a rebuild. |
| **Built the index into the Docker image** instead of at startup | No downloads on start, and each image ships the exact index it was tested with. |
| **800-char chunks + code-aware HTML cleaning** instead of bigger chunks or contextual headers (tested) | Both alternatives made answers worse. The cleaning mattered most: YAML used to come out one token per line. |
| Added **query rewrite ("code HyDE") + rank fusion** instead of plain vector search (tested) | Questions are prose but answers are YAML. The right Compose example ranked 26th with plain search. |
| Used an **LLM reranker** instead of a cross-encoder (tested) | The cross-encoder made the target question worse. Costs ~3 s more per answer, mostly hidden by streaming. |
| **10 reranked picks, no per-page cap** instead of capping picks per page (tested) | The cap cut the build commands out of the Dockerfile answer. |
| Added **rule-based context completion** instead of a bigger top-k or more LLM calls | Each rule fixes a real failure: missing `.env` values, a Dockerfile written from memory, Linux-only install steps. Free, fast and testable. |
| **Strict "docs only" prompt** instead of letting the model use its general knowledge | In DevOps a confident wrong command is worse than "the docs don't show this". Answers are sometimes less helpful on purpose. |
| **Requires checklist** after each config example | Answers kept leaving out what an example needs: variables, files, run flags, versions and feature gates. |
| **Own eval suite with deterministic checks** instead of an LLM judge alone | The judge rated invented YAML 5/5. Deterministic checks are cheap and say exactly what failed. 26 cases, every live bug added as one. |
| **Live grounding check that warns** instead of blocking or regenerating | Unverified code is often correct, and regenerating doubles cost and latency. False alarms were fixed with regression tests. |
| **Streaming as NDJSON over `fetch`** instead of Server-Sent Events | `EventSource` can't send a POST body. Sources appear before the first word, and errors arrive as an event. |
| **Plain HTML/CSS/JS** instead of React | One page with a text box: no build step, no extra dependencies. Revisit if it needs chat history or accounts. |
| **In-memory rate limits** instead of Redis (for now) | Every question is a paid API call on a public app, and no extra service was needed. Limits are per instance, so 2 instances can mean 2× the limit. |
| **Trust only the proxy-added IP** in `X-Forwarded-For` instead of the first entry | Clients can fake the first entry. Verified on the live service with spoofed headers. |
| **Multi-stage image, CPU-only PyTorch, non-root user** | About 3 GB of unused CUDA libraries removed and a smaller attack surface. The image is still ~600 MB. |
| Built **`chroma-hnswlib` with `HNSWLIB_NO_NATIVE=1`** | Without it, the library was compiled for the CI machine's exact CPU and crashed silently on other machines. |
| **Compose with one service** instead of several containers | The app is one process: Chroma is embedded and the LLM is an API. Becomes multi-container when a shared Redis is needed. |
| **Cloud Run with scale to zero, max 2 instances** instead of a VM or GKE | Near-zero cost for a portfolio app with a capped worst case. The first visitor after a quiet period waits for a cold start. |
| **HTTP startup probe on `/health`** instead of Cloud Run's default TCP check | The cold start can take longer than the default check allows. A broken revision fails the deploy, and the old one keeps serving. |
| **Workload Identity Federation** instead of a service-account JSON key | No long-lived credential stored in GitHub that could leak. |
| **Anthropic key in Secret Manager** instead of a GitHub secret | The key never touches GitHub or the image. |
| **Tests + Docker smoke test gate every deploy** | A broken build or a container that won't start never reaches production. |
