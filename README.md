# HTML Docs RAG

Citation-aware RAG over multiple Knowledge Bases built from HTML documentation

HTML Docs RAG is a proof of concept for preparing static HTML documentation,
building local retrieval indexes, and answering questions with citations. It
supports multiple Knowledge Bases, selects loaders and parsers through TOML,
and exposes both a Python CLI and a read-only REST API.

The evaluated documentation sets are:

- the official Japanese Python 3.13 documentation, parsed with a dedicated
  Python/Sphinx parser; and
- the official uv Documentation, parsed with the configurable generic HTML
  parser.

Embedding, reranking, and generation use local Hugging Face models. The
application runtime does not require the OpenAI API, and the final local quality
sprints did not use the OpenAI API or OpenAI Judge. An optional evaluation
helper and historical OpenAI Judge results are documented in
[`evaluation/rag_quality_summary.md`](evaluation/rag_quality_summary.md); those
historical evaluations require separate API access to reproduce. Model weights,
generated indexes, processed corpora, and caches are not included.

## What it provides

- Citation-aware answers whose source URLs come from retrieved metadata, not
  model-generated URLs.
- A strict answer-or-abstain contract in addition to the backward-compatible
  legacy answer mode.
- Multiple independently prepared Knowledge Bases served by one runtime.
- TOML-selected loader and parser combinations with fail-closed validation.
- A dedicated parser for Python's Sphinx-generated documentation.
- A configurable parser for conventional static HTML documentation.
- Pinned local ZIP archives, source-locked mutable ZIP archives, bounded static
  HTML crawling, and local expanded HTML trees.
- Local Hugging Face embedding, reranking, and generation models.
- A read-only REST API for querying prebuilt Knowledge Bases.
- Reproducible source acquisition through a frozen snapshot, provenance record,
  and source lock for mutable upstream archives.

The shared preparation flow is:

```text
source -> loader -> parser -> sections -> chunks -> indexes -> RAG -> cited answer
```

## Compatibility names

The public repository is named `html-docs-rag`, while the Python distribution
name remains `python-doc-rag-assistant`.

The `python_doc_rag` import namespace is retained for compatibility with the
original Python-document PoC. The current architecture supports multiple
Knowledge Bases built from HTML documentation.

These interfaces are intentionally unchanged:

```python
import python_doc_rag
```

```bash
python -m python_doc_rag --help
```

## Runtime profiles and deployment scope

| Selection | Retrieval | Reranker | Generator and answer contract |
| --- | --- | --- | --- |
| No `--profile` | Dense | None | `Qwen/Qwen3-4B-Instruct-2507` with no profile revision pin; legacy answer mode |
| `recommended-v1` | Hybrid candidates | mMARCO MiniLM | `Qwen/Qwen3-4B-Instruct-2507`; `answer-or-abstain-v1` |
| `recommended-v2` or `recommended` | Technical-field retrieval with BGE-M3 | mMARCO MiniLM | `Qwen/Qwen3-8B`; `answer-or-abstain-v1` |

`recommended` currently aliases `recommended-v2`. `recommended-v1` remains a
lower-memory rollback option, while omitting `--profile` preserves the original
Dense, no-reranker, legacy-answer defaults.

The recorded `recommended-v2` runs used an NVIDIA L4 with 23,034 MiB of VRAM;
the profile targets a 24 GB-class GPU and is not guaranteed to fit a 16 GB-class
environment. GPU memory and latency are environment-specific reference values.

The supported inputs are bounded static HTML documentation sources. JavaScript
rendering, authenticated sites, PDF/Word ingestion, and universal crawling are
out of scope. The REST API is a trusted-environment, single-worker PoC without
authentication or authorization; it is not intended as a production API on the
public Internet.

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)

The only dependency source of truth is `pyproject.toml` together with
`uv.lock`; no separate dependency list is maintained.

Create the lightweight development environment from the locked dependencies:

```bash
uv sync --frozen
```

Install local model and FAISS dependencies when preparing data or answering
questions:

```bash
uv sync --frozen --extra inference
```

Add the API dependencies when running the service:

```bash
uv sync --frozen --extra inference --extra api
```

Google Colab uses the same frozen lock while excluding development dependencies:

```bash
uv lock --check
uv sync --frozen --extra inference --no-dev
```

Include the API extra only when it is needed in Colab:

```bash
uv sync --frozen --extra inference --extra api --no-dev
```

## CLI

The package provides six commands:

- `ask`: answer one question.
- `chat`: answer multiple independent questions while reusing loaded models.
- `check`: validate prepared artifacts without loading models.
- `profile`: display a fixed runtime profile without loading models.
- `prepare`: acquire and prepare an HTML documentation data set.
- `serve`: run the read-only multi-KB REST API.

Inspect any command without downloading a model:

```bash
uv run --frozen python -m python_doc_rag --help
uv run --frozen python -m python_doc_rag prepare --help
uv run --frozen python -m python_doc_rag check --help
uv run --frozen python -m python_doc_rag profile --help
uv run --frozen python -m python_doc_rag ask --help
uv run --frozen python -m python_doc_rag chat --help
uv run --frozen python -m python_doc_rag serve --help
```

## Source modes and parsers

Loader and parser selection is explicit in each site TOML. A parser is not
guessed from a loader type.

| Example | Loader | Parser | Source behavior |
| --- | --- | --- | --- |
| Frozen Python docs | `pinned-local-archive` | `python-sphinx` | Verifies and reads the repository snapshot |
| Current Python docs | `snapshot-http-archive` | `python-sphinx` | Downloads once, records a source lock, and refreshes only when requested |
| Expanded Python docs | `local-html-tree` | `python-sphinx` | Reads a caller-supplied local HTML tree |
| uv Documentation | `bounded-http` | `generic-html` | Crawls bounded static HTML under configured origins and paths |

The Python parser preserves the established handling of Sphinx structure,
permalinks, FAQ pages, headings, and navigation noise. The generic parser is
configured with CSS selectors, heading levels, minimum text length, and fallback
behavior in TOML; it does not contain uv-specific parsing code.

The bounded HTTP loader limits origins, path prefixes, page count, response
size, redirects, retries, and request pacing. It respects `robots.txt`, rejects
non-HTML responses, and supports offline replay of its validated raw cache.

Implementation and validation details are in
[`docs/source_snapshot_and_refresh_policy.md`](docs/source_snapshot_and_refresh_policy.md)
and
[`evaluation/configurable_html_ingestion_summary.md`](evaluation/configurable_html_ingestion_summary.md).

## Frozen Python documentation example

The frozen Python site configuration uses the included source snapshot and does
not fetch Python HTML from the network:

Every `/path/to/...` value below is a placeholder. Replace it with a writable
location outside the repository and reuse the same prepared data roots in any
service configuration.

```bash
uv run --frozen --extra inference \
  python -m python_doc_rag prepare \
  --site-config configs/sites/python-docs.toml \
  --data-root /path/to/python-docs-data \
  --device cuda
```

Validate the generated artifacts before asking a question:

```bash
uv run --frozen --extra inference \
  python -m python_doc_rag check \
  --data-root /path/to/python-docs-data \
  --profile recommended-v2
```

```bash
uv run --frozen --extra inference \
  python -m python_doc_rag ask \
  --data-root /path/to/python-docs-data \
  --profile recommended-v2 \
  --question "list.sort()がNoneを返すのはなぜですか？"
```

These commands select `recommended-v2` explicitly. Use `recommended-v1` for the
retained lower-memory rollback configuration; omitting `--profile` preserves the
original Dense, no-reranker, legacy-answer defaults.

## Source snapshot reproducibility

The repository includes one reviewed binary exception:

```text
resources/source_snapshots/python-3.13-ja-2026-07-20.zip
```

- Project snapshot SHA-256:
  `1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777`
- Size: `17,310,566` bytes
- Members: `1,258`
- Protected corpus: `384` pages, `2,766` sections, `8,677` chunks
- Protected chunk SHA-256:
  `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869`

The upstream archive URL is a mutable alias. Its acquisition record contains
the original recorded archive SHA, while the repository ZIP is a deterministic
project snapshot reconstructed from the retained expanded tree. The two hashes
are deliberately distinct and are not represented as identical artifacts.

See the machine-readable
[`provenance record`](resources/source_snapshots/python-3.13-ja-2026-07-20.provenance.json),
the
[`snapshot validation summary`](evaluation/python_docs_source_snapshot_summary.md),
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Mutable archive and local-tree modes

The current Python configuration downloads the mutable upstream ZIP only when
needed and writes its SHA-256, byte size, requested/final URL, and acquisition
time to `data/raw/source.lock.json`:

```bash
uv run --frozen --extra inference \
  python -m python_doc_rag prepare \
  --site-config configs/sites/python-docs-current.toml \
  --data-root /path/to/python-current-data \
  --device cuda
```

Use `--offline` to replay the locked local source without a network request and
`--refresh` only when intentionally replacing that source. Use `--rebuild` to
recompute downstream artifacts from an already validated source.

The compatibility configuration for an expanded local Python HTML tree is:

```bash
uv run --frozen --extra inference \
  python -m python_doc_rag prepare \
  --site-config configs/sites/python-docs-local-compat.toml \
  --source-root /path/to/python-3.13-docs-html \
  --data-root /path/to/python-local-data \
  --device cuda
```

## Generic HTML example: uv Documentation

The uv portability configuration uses bounded crawling and the generic HTML
parser:

```bash
uv run --frozen --extra inference \
  python -m python_doc_rag prepare \
  --site-config configs/sites/uv-docs-smoke.toml \
  --data-root /path/to/uv-docs-data \
  --device cuda
```

This is a bounded portability demonstration, not a universal crawler. Site
coverage is controlled by the checked-in start URLs, crawl limits, and parser
selectors.

## Citation and answer contracts

The RAG pipeline selects retrieved chunks within an explicit token budget and
assigns citation labels only after selection. The generator does not receive
source URLs. Final source links are assembled from trusted retrieval metadata.

The citation finalizer rejects malformed, missing, or out-of-range citations,
URLs in generated answer text, and Markdown links. It performs at most one
repair generation with the same selected evidence and otherwise fails closed.

The opt-in `answer-or-abstain` mode requires one strictly validated JSON object:

- `answer` requires non-empty answer text and at least one valid citation.
- `abstain` returns no answer text or sources and uses a fixed reason code.
- zero retrieval results abstain without calling the generator.
- invalid output receives one repair attempt and then fails closed.

The legacy citation-aware mode remains the default for compatibility. The
answer-or-abstain implementation is experimental and is not presented as a
general correctness guarantee. See
[`evaluation/answerability_contract_summary.md`](evaluation/answerability_contract_summary.md).

## Multi-KB REST API

Prepare and validate every Knowledge Base before starting the service. The
service does not upload, modify, delete, or reindex data while running.

The checked-in `configs/services/multi-kb.example.toml` is a template. Its
relative `data_root` values are placeholders resolved from that file's
directory. Copy it to a local path outside the repository and replace both
values with the data roots produced by `prepare`. The template contains:

```toml
revision = "multi-kb-service-v1"
profile = "recommended-v2"
device = "cuda"

[[knowledge_bases]]
id = "python-docs"
display_name = "Python 3.13 日本語公式ドキュメント"
data_root = "../../example-data/python-docs"

[[knowledge_bases]]
id = "uv-docs"
display_name = "uv Documentation"
data_root = "../../example-data/uv-docs"
```

For example, replace the two values above with
`/path/to/python-docs-data` and `/path/to/uv-docs-data`. The service starts only
after every configured Knowledge Base passes validation.

Start the service on its loopback default:

```bash
uv run --frozen --extra inference --extra api \
  python -m python_doc_rag serve \
  --service-config /path/to/local-multi-kb.toml \
  --host 127.0.0.1 \
  --port 8000
```

HTTP endpoints are:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Process liveness |
| `GET` | `/readyz` | Full runtime readiness |
| `GET` | `/v1/knowledge-bases` | Safe metadata for registered Knowledge Bases |
| `POST` | `/v1/knowledge-bases/{knowledge_base_id}/answers` | Answer one independent question against one Knowledge Base |

Example request:

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{"question":"uv syncは何を行うコマンドですか？"}' \
  http://127.0.0.1:8000/v1/knowledge-bases/uv-docs/answers
```

One shared embedding model, reranker, and generator are loaded for the process;
retrieval artifacts remain isolated per Knowledge Base. Startup is
all-or-nothing. The supported service topology is one Uvicorn worker, and a
global semaphore serializes answer processing while concurrent requests wait.
Full details are in
[`docs/multi_knowledge_base_api.md`](docs/multi_knowledge_base_api.md) and the
[`API smoke summary`](evaluation/multi_knowledge_base_api_smoke_summary.md).

## Feature status

- Implemented runtime paths include generic HTML ingestion, the Python Sphinx
  parser, source locking, frozen snapshot ingestion, uv portability,
  `recommended-v1`, `recommended-v2`, the legacy answer mode, the
  `answer-or-abstain-v1` contract, and the multi-KB REST API.
- Existing-chunk Parent Retrieval, Full Section Parent, combined reranker plus
  parent retrieval, evidence-first generation, and two-stage generation are
  research-only implementations or evaluation runners. None is part of the
  current CLI/API runtime; evaluations rejected them or did not select them.
- The OpenAI Judge helper is optional evaluation tooling with historical results;
  it is not an application-runtime dependency or production decision path.
- Authentication, online reindexing, JavaScript rendering, PDF/Word ingestion,
  and multi-worker or multi-GPU production deployment are unsupported.

## Evaluation scope

The repository contains checked-in question sets and human-readable summaries,
not raw model outputs or generated indexes. The recorded evaluations cover:

- Python 3.13 Japanese documentation retrieval and cited generation;
- uv Documentation ingestion and retrieval portability;
- dense, hybrid, technical-field, reranked, and parent-retrieval experiments;
- answerability and answer-or-abstain behavior; and
- multi-KB API behavior.

The currently selected profile and its limitations are documented in
[`evaluation/final_quality_sprint_v2_summary.md`](evaluation/final_quality_sprint_v2_summary.md).
These results are specific to the checked-in configurations and question sets;
they are not evidence of universal HTML-site or question-answering support.

## Explicit non-goals and unsupported cases

This proof of concept does not support:

- arbitrary JavaScript rendering;
- authenticated websites;
- PDF or Word ingestion;
- universal web crawling;
- online reindexing through the REST API;
- API authentication or authorization;
- multi-worker production deployment; or
- multi-GPU production deployment.

The API is intended for a trusted environment. It has no API key, user model,
ACL, or production network-security layer. Binding outside loopback requires an
independently managed network boundary.

## Data and artifact policy

The following are intentionally excluded from Git:

- local data roots and raw mutable-source caches;
- processed corpora and embedding outputs;
- FAISS and other generated indexes;
- model weights and Hugging Face caches;
- experiment raw outputs; and
- generated reports and notebooks.

The frozen Python documentation ZIP is the sole reviewed source-snapshot
exception. `.gitignore` contains the corresponding local artifact and cache
rules. Tests use small fixtures and do not download models.

## Development

Run the lightweight validation suite:

```bash
uv lock --check
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Validate every optional dependency group:

```bash
uv sync --frozen --all-extras
uv run --frozen --all-extras pytest -q
uv run --frozen --all-extras ruff check .
```

Repository layout:

```text
configs/                     Site and multi-KB service TOML
docs/                        Public technical design notes
evaluation/                  Question sets and evaluation summaries
resources/source_snapshots/  Frozen source snapshot and provenance
scripts/                     Reproducible preparation/evaluation utilities
src/python_doc_rag/           Package implementation
tests/                        Fixture-based tests
```

## License

Original project code is licensed under the MIT License. The included Python
documentation snapshot remains subject to its upstream license and copyright
notices. Local Hugging Face models are subject to each model's own license.
Project licensing does not replace or override third-party terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
