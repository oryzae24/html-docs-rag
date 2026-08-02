# Python documentation source snapshot validation summary

## Scope and status

This document records the measured frozen/current source validation on branch
`feat/pinned-python-doc-archive-ingestion`, based on main
`8e7508b98cd2dae47d194f3b246108d267b1e8bd`.
The independent pre-merge review started from PR head
`92fc8c59a496b26f866b68bf7e1d2ad36560ea8c`. The final reviewed commit is the
commit containing this document and is reported in the draft PR because a
commit cannot embed its own SHA.

The repository snapshot, protected-corpus reproduction, complete fresh GPU
builds, current-source lifecycle, uv regression, and frozen Python + uv API
smoke were executed on 2026-08-01. The standard/all-extra test totals are
recorded in the validation matrix below. Actual Google Colab was not used for this
branch validation; the GPU integration runs below used a RunPod NVIDIA L4.

## Independent pre-merge review

The review exercised source identity, publication and rollback as one combined
state machine rather than relying on the implementation report. No protected
baseline was changed. The evidence-backed findings fixed during the review were:

- **High:** pending publication could expose a mixed source/derived generation,
  and a partial rollback could remain marked committed and later retire the only
  valid backup. Runtime gating, durable rollback phase changes, and strict
  all-generation recovery now prevent forward-finalizing mixed artifacts.
- **High:** cleanup and recovery interrupt windows could roll a valid committed
  generation backward, leave a successful generation unnecessarily gated, or
  accept a current candidate with mismatched fingerprints or a downgraded legacy
  manifest. Recovery is now idempotent and requires the current v2 manifest plus
  exact source and processing identities before retiring backups.
- **High:** archive download/extraction reopened security-sensitive paths after
  verification. A concurrent local path replacement could redirect retry writes
  or change the archive/selected HTML inode. The transport and ZIP loader now
  keep nofollow descriptors and verify the same inode/content throughout.
- **High:** Windows junctions and managed-path aliases were not rejected at all
  source snapshot, cache, staging, and publication boundaries. They are now
  rejected before a read, write, rename, or deterministic snapshot traversal.
- **Medium:** an otherwise recoverable committed-source/corrupt-derived state had
  no explicit repair path. Explicit rebuild/refresh-resume now quarantines the
  old journal/staging as a recoverable bundle and publishes only a fully
  validated replacement.
- **Medium:** same-byte current refresh, bounded raw-only replay, stale profile
  resume, extraction-manifest tampering, central-directory preflight, and
  deterministic source-tree mutation/output-swap cases had incomplete recovery
  or validation. Each now has a failure-injection regression test.
- **Medium:** legacy configuration/loading facades and the no-argument Python
  parser adapter had compatibility regressions, including Python-version metadata
  and citation-base behavior. The legacy paths are restored without changing the
  explicit new parser settings contract.

No blocker remained: malicious ZIP inputs did not obtain a remote arbitrary-path
write primitive, snapshot/provenance values stayed truthful, and the protected
corpus remained byte-identical. No unresolved high or medium finding remains.
The only retained low item is disk-space cleanup: an interrupted cleanup can
leave a safe content-addressed cache or recovery bundle. It is never treated as
a completed dataset and can be inspected before manual removal. Deterministic
regeneration was exact in the two tested locale/timezone environments.

## Original acquisition record

The retained metadata records the following original acquisition:

| Field | Recorded value |
| --- | --- |
| Requested URL | `https://docs.python.org/ja/3.13/archives/python-3.13-docs-html.zip` |
| URL policy | Mutable Python 3.13 Japanese documentation alias |
| Downloaded at | `2026-07-20T20:34:21.446060+00:00` |
| Recorded SHA-256 | `f8ddb3454726cbe34580b4c21723128a1b33b50f1155e9b9184cb790db66d9cb` |
| Recorded byte size | 17,367,739 |
| Final URL | Not recorded |
| Original archive bytes | Unavailable when the project snapshot was created |

The recorded original SHA is provenance, not the identity of the ZIP committed
to this repository. The mutable upstream currently cannot be assumed to return
those original bytes.

## Deterministic project snapshot

The expanded HTML tree retained from the acquisition reproduced the protected
chunk corpus byte-for-byte. That validated tree was then packaged without
changing file contents using the deterministic project snapshot builder.

| Field | Value |
| --- | --- |
| Snapshot path | `resources/source_snapshots/python-3.13-ja-2026-07-20.zip` |
| Snapshot SHA-256 | `1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777` |
| Snapshot byte size | 17,310,566 |
| Archive format | ZIP |
| Member count | 1,258 |
| Archive root | `python-3.13-docs-html` |
| Source base URL | `https://docs.python.org/ja/3.13/` |
| Python parser version | `3.13` |
| Original archive bytes represented as identical | No |

The full derivation, source tree hash, deterministic ZIP settings, file and
directory counts, and license/copyright paths are recorded in
`resources/source_snapshots/python-3.13-ja-2026-07-20.provenance.json`.
The documentation license and copyright files remain inside the snapshot.

The final snapshot writer was run twice against the same retained tree: once
with `TZ=Pacific/Honolulu`, `LC_ALL=C`, and once with `TZ=Asia/Tokyo`,
`LC_ALL=C.utf8`. Both outputs were 17,310,566 bytes and had SHA-256
`1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777`,
byte-identical to the committed ZIP. The source tree SHA-256 was
`fbed1fed2ad21e4f9eee2ce8d2280fc4140959d67de00df03f6f897765ce0a3b`.

## Protected corpus reproduction

The retained expanded tree was processed with the protected Python/Sphinx
parser and chunking behavior before the project snapshot was created.

| Metric | Protected value | Observed result |
| --- | ---: | --- |
| Selected source pages | 384 | 384 |
| Sections | 2,766 | 2,766 |
| Chunks | 8,677 | 8,677 |
| Chunk JSONL SHA-256 | `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869` | Same |
| Byte content and JSONL line order | Protected baseline | Byte-identical |

The same result was then reproduced from the committed project ZIP in a fresh
data-root by the standard one-command `prepare`: its chunk JSONL was also
byte-identical to the protected baseline.

## Implemented source architecture

The standard frozen configuration is
`configs/sites/python-docs.toml` with
`pinned-local-archive + python-sphinx`. The independently refreshable current
configuration is `configs/sites/python-docs-current.toml` with
`snapshot-http-archive + python-sphinx`. The expanded-tree compatibility route
is `configs/sites/python-docs-local-compat.toml` with
`local-html-tree + python-sphinx`.

Loader and parser settings are discriminated independently and resolved by the
built-in registries. Common ingestion receives `SourceDocument`, invokes the
configured parser, and passes the resulting sections to shared chunk/index/RAG
processing. Python/Sphinx extraction is isolated under
`src/python_doc_rag/sites/python_docs/`; generic HTML parsing is isolated under
`src/python_doc_rag/parsers/`. uv remains
`bounded-http + generic-html` and has no dedicated parser.

The mutable archive path records its first observed archive in
`data/raw/source.lock.json`. The lock contains requested/final URLs, observed
SHA-256, byte size, archive/root/base URL settings, acquisition time,
`source_config_sha256`, source snapshot identity, and a data-root-relative
content-addressed cache path. Normal reuse does not query upstream;
`--refresh` is the only source-update operation. `--rebuild` uses only a local
validated source and changes no source lock. `--offline` prohibits document
source network access.

Four identities have distinct roles:

- `source_config_sha256`: source-affecting loader settings
- `processing_config_sha256`: dataset/parser/chunk/index/profile settings
- `source_snapshot_sha256`: actual archive bytes or ordered portable page
  content identity
- `site_config_file_sha256`: raw TOML bytes retained for audit

## Validation matrix

| Validation | Status | Recorded result |
| --- | --- | --- |
| Recorded original SHA/size provenance | Complete | Metadata values recorded above |
| Original ZIP bytes located | Not available | The original bytes were not present when the project snapshot was made |
| Repository snapshot file identity | Complete | 17,310,566 bytes; SHA-256 `1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777` |
| Repository snapshot ZIP inventory | Complete | 1,258 members; root `python-3.13-docs-html` |
| Deterministic ZIP regeneration | Complete | Two runs under different timezone/locale settings were byte-identical to the repository ZIP |
| Expanded tree vs protected corpus | Complete | 384 pages, 2,766 sections, 8,677 chunks; byte-identical protected JSONL |
| Frozen fresh `prepare` from repository ZIP | Complete | 384 pages, 2,766 sections, 8,677 chunks; protected JSONL byte-identical |
| Frozen Dense FAISS count/dimension | Complete | 8,677 records / 384 dimensions |
| Frozen BGE-M3 artifact and symbol sidecar | Complete | 8,677 records / 1,024 dimensions; 8,677 symbol records |
| Frozen `check --profile recommended-v2` | Complete | Validation passed |
| Frozen `ask` citation/answer smoke | Complete | `answer`; official `docs.python.org` FAQ citation; no local path |
| Completed dataset second-run reuse | Complete | Dataset, source manifest, index, and profile artifacts reused without regeneration |
| Frozen `--rebuild` | Complete | Local project snapshot only; source SHA unchanged; protected chunk SHA reproduced |
| Expanded-tree local compatibility route | Complete | `--source-root` produced 384 pages, 2,766 sections, 8,677 chunks, and the protected chunk SHA |
| Current first fetch and observed SHA/size/counts | Complete | 17,379,980 bytes; SHA/counts recorded below |
| Current second run with zero HTTP calls | Complete | Completed dataset/local lock reused; source lock unchanged |
| Current raw-only offline replay | Complete | Rebuilt from copied lock and archive cache with source network disabled |
| Current explicit refresh | Complete | Explicit download succeeded; observed archive bytes remained the same |
| uv 31-page/184-section/270-chunk regression | Complete | Counts and protected portability chunk SHA reproduced |
| uv cache reuse/offline/refresh/check/ask | Complete | Normal reuse, raw-only offline replay, explicit refresh, check, and standalone ask passed |
| Frozen Python + uv multi-KB API smoke | Complete | Two KBs, domain isolation, abstain/404 contracts, shared models, clean exit |
| Transaction/ZIP targeted suite | Complete | 308 passed; rollback, interrupt, cache tamper, path replacement, and ZIP-limit injections included |
| Standard pytest and Ruff | Complete | 735 passed, 18 skipped in 82.84 s; Ruff passed |
| All-extras pytest and Ruff | Complete | 753 passed in 100.18 s; Ruff passed |
| Lock and lightweight CLI help | Complete | `uv lock --check`, frozen syncs, and four CLI help commands passed |
| GPU device and peak VRAM for this review | Complete | NVIDIA L4, 23,034 MiB total; 19,666 MiB observed during the final API smoke |
| Actual Google Colab run | Not executed | RunPod L4 clean-room smoke completed; no actual Colab claim |

The actual Colab run must not be inferred from this Linux/RunPod clean-room
smoke. Historical Colab compatibility results elsewhere in the repository are
separate experiments and do not constitute a Colab run of this source workflow.

## Frozen end-to-end results

The fresh frozen build read the committed project snapshot with no Python
document-source network access. The project snapshot identity remained
`1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777`;
it was not replaced by the original recorded SHA.

| Artifact | Records / dimension | SHA-256 |
| --- | --- | --- |
| Chunk JSONL | 8,677 | `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869` |
| Baseline Dense FAISS | 8,677 / 384 | `0a60928abd5d541321c3830bd55332f1df9e7fb1aec1c007d1f636d711d9d45b` |
| BGE-M3 FAISS | 8,677 / 1,024 | `96c45fb2a3cd3c545792fca4cab15fcac71be5a64750418131ff2bc9ec71e090` |
| Symbol sidecar | 8,677 | `15dc7f8c9d83a16a91ffbf11dc9015b4a5ce6f71545fb81837f3babbc8545c1a` |

`check --profile recommended-v2` passed. The question
`list.sort()がNoneを返すのはなぜですか？` returned `status=answer` with
`https://docs.python.org/ja/3.13/faq/design.html#why-doesn-t-list-sort-return-the-sorted-list`;
no local path appeared in the result. A normal second `prepare` reused the
completed dataset without changing source-manifest or index/profile mtimes. A
full `--rebuild` used the same local source snapshot, preserved its SHA, and
reproduced the protected chunk SHA and complete artifacts.

## Current mutable-alias results

On 2026-08-01, the current configuration acquired and locked these independently
observed upstream bytes:

| Field | Observed value |
| --- | --- |
| Requested URL | `https://docs.python.org/ja/3.13/archives/python-3.13-docs-html.zip` |
| Final URL | `https://docs.python.org/ja/3.13/archives/python-3.13-docs-html.zip` |
| Archive SHA-256 | `cf886e2221121ceef3353cc0f78714ad4c4619092aec5a9bddb020cef77122c0` |
| Archive byte size | 17,379,980 |
| Source pages / sections / chunks | 384 / 2,770 / 8,694 |
| Chunk JSONL SHA-256 | `0da3294898136f4674243a721aaf3015cfe59433eb236fe3d3768a845f3b7e62` |

These current bytes are neither the original recorded archive identity
`f8ddb3454726cbe34580b4c21723128a1b33b50f1155e9b9184cb790db66d9cb`
nor the deterministic project snapshot identity
`1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777`.
They are locked only for the current data-root and are not a protected-baseline
update.

| Artifact | Records / dimension | SHA-256 |
| --- | --- | --- |
| Baseline Dense FAISS | 8,694 / 384 | `eef9c1ddf3a9c6e16eb7502fd0534140f506d5e65b4cfb9bcdcbba6a0c914322` |
| BGE-M3 FAISS | 8,694 / 1,024 | `a1da00d410ea8867447a02fc13e819859a5c9d753b6a6bc71ff30cf99788969d` |
| Symbol sidecar | 8,694 | `6a38cb06fe08732a6fb8e1b077bbfe1957d21e4fd61fb73a326534f0513d956d` |

The normal second run reused the completed dataset without an HTTP request. A
separate empty data-root containing only the copied source lock and cached ZIP
rebuilt the corpus successfully with `--offline`. Explicit `--refresh` was the
only run that downloaded again; it observed the same archive SHA and published
the refreshed lock/downstream state safely. `check --profile recommended-v2`
passed. The same `list.sort()` question returned `status=answer`, page title
`Python 3.13.14`, and the official FAQ URL above, with no local path (retrieval
0.362 s, generation 6.596 s, total 6.995 s).

## uv and multi-KB regression results

The uv path remained `bounded-http + generic-html` and reproduced 31 pages,
184 sections, and 270 chunks.

| Artifact | Records / dimension | SHA-256 |
| --- | --- | --- |
| Chunk JSONL | 270 | `83b49de1a8b465f0e8e8c0f20961fdffa9b0c3b92dc76bda76a303a75763dc84` |
| Baseline Dense FAISS | 270 / 384 | `152220c72d95a0f4235ec54ef4866d0b32cc5c3348e708b81319f00ad249fb1a` |
| BGE-M3 FAISS | 270 / 1,024 | `38a5fb7fd117270db613ac7d0fefcc8421a9210cf7500d8689ca1a8fcc28e124` |
| Symbol sidecar | 270 | `5a9ac53294f50e70fd2a3036c8970818cea474f5ed70adcbf8be2576c04129dd` |

Normal reuse, a raw-cache-only offline replay, explicit refresh,
`check --profile recommended-v2`, and a standalone `ask` all passed. The
initial raw source snapshot SHA was
`83aae22c14e27108ff287c2763fb0a6b396fe8bcae6f111d9e34a635af303b1e`;
the implementation validation refresh produced
`ade870b0c0dd1e8195402b1c980c9453d692056961067f2346ab1a9425b4c122`,
and the independent review refresh later produced
`60385129a16b66259562e4cd38d047805414ae729655f876c5c7f484bc5b3d5f`.
The only raw changes were Cloudflare `data-cfemail` obfuscation values on three
pages (`guides/install-python/`, `guides/projects/`, and `guides/tools/`); the
parsed 270-chunk JSONL and its SHA remained unchanged.

The API smoke registered the frozen Python dataset and uv dataset. `readyz`
returned 200 with two knowledge bases. Python -> uv -> Python requests stayed
within their respective `docs.python.org` and `docs.astral.sh` URL domains;
unknown KB returned 404 and an out-of-corpus request returned 200 `abstain`.
The shared embedding, reranker, and generator/tokenizer loaders each ran once.
The final independent run used the freshly rebuilt frozen root, observed
19,666 MiB GPU memory at the high-water sample, had no OOM, and Ctrl+C produced
exit code 0 with a complete application shutdown.

## Automated validation

`uv lock --check` passed. The final standard environment completed
`uv sync --frozen`, then pytest with 735 passed and 18 skipped in 82.84 seconds,
and Ruff with no findings. The final all-extras environment completed
`uv sync --frozen --all-extras`, then pytest with 753 passed in 100.18 seconds,
and Ruff with no findings. The final targeted source/transaction/ZIP suite had
308 passing tests. Lightweight help succeeded for the package root and the
`prepare`, `check`, and `serve` subcommands.

## Security and provenance observations

The repository loader requires the project snapshot SHA before extraction.
Shared ZIP extraction rejects traversal and absolute paths, Windows/backslash
escapes, NULs, symlinks, special entries, normalized duplicate paths,
Windows reserved/oversized components, implicit-parent/casefold/file-directory
collisions, a reserved extraction-manifest path, configured declared/streamed
size limits, central-directory count/size inconsistencies, and a missing fixed
archive root. Cache reuse also rejects pre-existing symlink/junction ancestors,
extra files, modified members/manifests, and archive or selected-HTML inode
replacement. Archive acquisition, verification, and extraction retain the same
nofollow file descriptors across the security boundary.

Failure injection covered interrupted download, corrupt content-addressed ZIP,
extraction failure, source-lock publication followed by parser/chunk/Dense/BGE
failure, each derived publication step, dataset-manifest publication, refresh
over an existing dataset, partial rollback failure, committed cleanup
interruption, mismatched/legacy candidate manifests, and a second interruption
during explicit recovery. The old lock, fetch manifest, dataset manifest,
chunks, indexes, and profile either remained the active generation or were
restored together. A partial state was never accepted as complete; retained
orphan caches/backups remained outside runtime resolution and the next explicit
prepare recovered successfully.

Fingerprint mutation tests confirmed that TOML formatting changes do not fetch,
parser/chunk changes require rebuild without source acquisition, index/profile
changes rebuild downstream only, and archive URL/root/include or bounded crawl
scope changes invalidate raw reuse. The full pinned/archive/bounded/local loader
matrix covered normal reuse, raw-only replay, offline, rebuild, refresh, resume,
flag conflicts, corrupt locks/caches, missing snapshots, extraction corruption,
and source-tree mutation with the documented CLI exit behavior. Common
preparation/protocol/generic/ZIP modules retained no Python-site import; loader
and parser factories remain independently keyed. Legacy v1/manifestless roots,
local-tree construction, profile aliases, check/ask/chat, and fake API startup
continued to pass without an implicit manifest migration.

Archive acquisition URLs and citation URLs are separate settings. The loader
constructs portable source URLs from the configured `source_base_url` and
logical HTML paths; the Python parser adds anchors to those trusted page URLs.
Local filesystem paths are not citation sources, and URLs are not generated by
the language model.

## Reproducibility and repository policy

The deterministic frozen source ZIP is the sole archive exception committed to
Git for protected-experiment reproducibility. Current archives, raw caches,
expanded trees, processed corpora, indexes, model caches, and detailed runtime
artifacts remain outside Git. Git LFS is not used.

`recommended-v2` targets a 24 GB-class GPU. A 16 GB-class environment can stop
preparation at the baseline index and use `recommended-v1`. Full build time,
disk use, and model availability remain environment-dependent.

No OpenAI API was used to acquire, package, parse, or compare this source. The
pipeline continues to use local/open models for retrieval and generation.

## Known limitations

- The project snapshot is derived from a validated expanded tree; it is not the
  unavailable original ZIP byte sequence.
- The current source uses a mutable alias and is reproducible only after its
  first successful acquisition creates a valid local source lock/cache.
- Only ZIP archives are supported; patch-version discovery, tar archives,
  incremental archive updates, and HTTP Range download resume are out of scope.
- Current-source content is not a replacement for the protected frozen
  baseline and does not automatically update evaluation values.
- One standalone uv `ask` launched concurrently with another GPU workload hit
  OOM; serial standalone retry succeeded. The serialized API smoke did not OOM.
  Concurrent multi-process or multi-GPU scheduling remains out of scope.
