# Multi-KB API GPU Smoke

## Scope and provenance

- Source code commit: `29d6bc27c0c441f80b6e21efdb82fd00ee0d3013`
- Executed at: 2026-08-01 UTC
- Profile: `recommended-v2`
- Knowledge Bases, in ServiceConfig order:
  - `python-docs`: Python 3.13 Japanese documentation, 8,677 chunks
  - `uv-docs`: uv Documentation corpus used for portability testing, 270 chunks
- Process model: one Uvicorn worker and one shared answer slot
- OpenAI API used: no
- API key or secret used: no

This was an operational portability smoke, not a new retrieval or answer-quality
evaluation. Existing profiles, models, questions, metrics, and protected artifacts were
not changed or regenerated.

## Environment

| Item | Observed value |
| --- | --- |
| GPU | NVIDIA L4 |
| Total VRAM | 23,034 MiB |
| Driver | 580.126.20 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| dtype | bfloat16 (`recommended-v2`) |
| NVIDIA process memory after the request sequence | 19,482 MiB |
| Process RSS after the request sequence | 2,802,412 KiB (about 2.67 GiB) |

The NVIDIA value is a point-in-time `nvidia-smi` process measurement, not a CUDA
allocator peak. Peak allocated/reserved values were not exposed by the read-only API
and are therefore not claimed. No GPU OOM occurred.

## Startup

The process started at `06:18:14Z`; readiness was observed after approximately 136
seconds. Startup output contained one weight-loading sequence for each fixed shared
resource, in the enforced order below:

1. BGE-M3 embedding model: one load
2. mMARCO MiniLM reranker: one load
3. Qwen3-8B generator: one load

The two FAISS/metadata/field/symbol retrieval graphs were then loaded separately in the
same process. Unit tests independently assert that each model loader is invoked once for
two Knowledge Bases and that the per-KB retriever objects are distinct.

## Request sequence

All requests below reached the same server PID. Internal timings exclude time waiting
for the global answer semaphore.

| Case | HTTP | Outcome | Retrieval | Generation | RAG total | HTTP wall |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `/healthz` | 200 | `ok` | - | - | - | 0.0032 s |
| `/readyz` | 200 | ready, profile `recommended-v2`, 2 KBs | - | - | - | 0.0018 s |
| KB list | 200 | Python then uv; no local paths | - | - | - | 0.0022 s |
| Python `list.sort()` | 200 | answer, 1 generation call | 0.4458 s | 6.0190 s | 6.5011 s | 6.5056 s |
| uv `uv sync` | 200 | answer, 1 generation call | 0.0839 s | 6.1788 s | 6.2832 s | 6.2855 s |
| Python `sqlite3` | 200 | abstain, 1 generation call | 0.1550 s | 2.0431 s | 2.2281 s | 2.2298 s |
| Unknown KB | 404 | `knowledge_base_not_found` | - | - | - | 0.0015 s |
| `/healthz` during generation | 200 | `ok`; event loop remained responsive | - | - | - | 0.0014 s |
| Corpus-external deployment policy | 200 | abstain, 1 generation call | 0.1216 s | 1.6716 s | 1.8148 s | 1.8163 s |

The `list.sort()` response cited only a Python URL:
`https://docs.python.org/ja/3.13/faq/design.html#why-doesn-t-list-sort-return-the-sorted-list`.
The `uv sync` response cited only a uv URL:
`https://docs.astral.sh/uv/guides/projects/#running-commands`.
The subsequent Python request was routed back to the Python service and abstained
without body or sources. No source from either Knowledge Base appeared in a
response for the other.

## Result and limitations

The smoke passed its operational criteria: both prepared datasets were validated, the API
became ready only after both KBs loaded, alternating routing stayed isolated, answer and
abstain schemas worked, unknown IDs returned the stable 404 code, and the process shut
down cleanly with exit code 0. A health request also completed while generation was
active, confirming that blocking RAG work did not stop the event loop.

This does not establish production security or throughput. Authentication,
authorization, CORS policy, multiple workers, multiple GPUs, cross-KB retrieval, hot
reload, and request-history storage remain intentionally out of scope. Generation is
serialized, so concurrent answers queue rather than run in parallel.
