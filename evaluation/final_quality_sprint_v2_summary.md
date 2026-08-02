# Final quality sprint v2 evaluation

## Scope and protected baseline

- Branch: `feat/final-quality-sprint-v2`
- Start commit and `python-doc-rag-recommended-v1` tag:
  `281a6a1d0ad7f73f3f3e9f3016b9573b4ac47e23`
- Start time: `2026-07-31T22:27:03Z`
- Phase A evaluation-definition commit:
  `ade3bef4ce2989f9b64ee57af693f04f266949fd`
- Protected processed corpus: 8,677 chunks, SHA-256
  `1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869`.
- The restored raw HTML was processed with the current production Parser and
  Chunker into 8,677 chunks. The rebuilt JSONL has the same SHA-256 and is
  byte-identical to the protected processed corpus.
- Development, Holdout, RAG quality, and Answerability question SHA-256 values
  remained `363ed4d5...f40`, `f71870a2...c36`, `585cc420...2e9`, and
  `d14734bc...8b6` respectively.
- Existing recommended-v1, Qwen revision, answer-or-abstain-v1, citation URL
  boundary, raw archive, questions, and baseline artifacts were not changed.
- OpenAI API and OpenAI Judge were not used. No API key or secret was read or
  stored.

Experiments were run on an NVIDIA L4 (23,034 MiB), driver `580.126.20`, PyTorch
`2.11.0+cu128`, CUDA runtime 12.8, Transformers 5.14.1, Sentence Transformers
5.6.1, and FAISS 1.14.3. Latency and memory are L4-specific reference values.

## Phase A: Candidate Recall Tournament

Revision: `technical-field-retrieval-v1` with
`candidate-recall-evaluation-v1`.

Candidate generation and reranking are measured separately. Recall@10/20/30,
first relevant rank, exact URL presence, distinct URL/page/section counts, and
candidate-generation time are recorded before the frozen recommended-v1
mMARCO MiniLM reranker is applied. A relevant candidate is identified only from
the evaluation question's expected URL keywords.

### Symbol and field-aware retrieval

The sidecar is generated generally from every parser-derived SearchChunk's
page title, section title, body, and trusted URL path/anchor. Python-shaped
identifiers preserve dotted qualification, underscores, numeric subscripts,
and call/no-call variants. Qualified suffixes and casefold variants are derived
deterministically; no evaluation API or hard case is encoded in the extractor.
The sidecar has 8,677 aligned records, 510,232 identifier-variant occurrences,
47,909 unique variants, size 7,365,599 bytes, and SHA-256
`15dc7f8c9d83a16a91ffbf11dc9015b4a5ce6f71545fb81837f3babbc8545c1a`.
Loading rejects count, order, chunk identity, text hash, or re-extraction drift.

The isolated ranks are identifiers, section title, page title, dense body, and
lexical body. Their scores are never added directly. Each field first produces
a rank; deterministic weighted RRF (`rrf_k=10`, field candidate k 30) performs
the fusion. SearchChunk metadata is returned unchanged, and source URLs remain
retrieval metadata only.

Only the four predeclared field settings were compared on Development:

| Field setting | Recall@10 | Recall@20 | Recall@30 | Rerank Hit@5 | Rerank MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal | 0.8846 | 0.8846 | 0.9231 | 0.9231 | 0.7147 |
| Identifier priority | 0.7308 | 0.8462 | 0.8846 | 0.8462 | 0.6821 |
| Title priority | 0.6154 | 0.8077 | 0.8462 | 0.8462 | 0.6891 |
| Identifier + title priority | 0.6538 | 0.8462 | 0.8846 | 0.8462 | 0.6901 |

Equal fusion was frozen. Increasing identifier/title weights crowded out useful
semantic or operational candidates and was rejected.

### Retrieval-specialized Embedding candidates

The bounded search stopped after two candidates. Both load through the existing
Sentence Transformers boundary with `trust_remote_code=False`, use normalized
embeddings, and have permissive licenses.

| Candidate | Pinned revision | Model-card basis | Dimension / max length / pooling | Prefix | License |
| --- | --- | --- | --- | --- | --- |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | [Over 100 languages and dense retrieval](https://huggingface.co/BAAI/bge-m3) | 1,024 / 8,192 / CLS | none | MIT |
| `intfloat/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` | [94 languages and asymmetric retrieval prefixes](https://huggingface.co/intfloat/multilingual-e5-base) | 768 / 512 / mean | `query: ` / `passage: ` | MIT |

The protected baseline index was not overwritten. BGE-M3 built in 388.50 s,
is 35,541,037 bytes, and has SHA-256
`96c45fb2a3cd3c545792fca4cab15fcac71be5a64750418131ff2bc9ec71e090`.
E5-base built in 127.27 s, is 26,655,789 bytes, and has SHA-256
`8d29646ab7c2a3c54ca3de5fc9d8d70ae3edab689645a77afff700ec9d74b7ec`.
Both metadata snapshots are byte-identical to the protected corpus SHA. During
evaluation, BGE-M3 plus the reranker used at most 2.89 GB CUDA allocated / 2.96
GB reserved and 3.62 GB process RSS; E5-base plus the reranker used 1.74 / 1.84
GB CUDA and 2.82 GB RSS. Observed BGE index-build process memory in `nvidia-smi`
ranged from 4,996 to 5,652 MiB; allocator and CPU peaks were not captured by the initial
atomic build, so this is explicitly an observed range rather than a peak claim.

### Development selection

| Candidate | Recall@10 | Recall@20 | Recall@30 | Rerank Hit@5 | Rerank MRR@10 | Candidate mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| recommended-v1 retrieval | 0.8846 | 0.8846 | 0.8846 | 0.8846 | 0.6667 | 0.0180 s |
| Equal field, baseline Embedding | 0.8846 | 0.8846 | 0.9231 | 0.9231 | 0.7147 | 0.0202 s |
| BGE-M3 only | 0.8462 | 0.9231 | 0.9231 | 0.8846 | 0.6370 | 0.0210 s |
| **Equal field + BGE-M3** | **0.8077** | **0.9615** | **0.9615** | **0.9615** | **0.7179** | **0.0308 s** |
| E5-base only | 0.8077 | 0.9615 | 0.9615 | 0.9615 | 0.7096 | 0.0154 s |
| Equal field + E5-base | 0.8077 | 0.9231 | 0.9231 | 0.9231 | 0.7051 | 0.0255 s |

Equal field + BGE-M3 was selected by the predefined order: Recall@30, reranked
Hit@5, then MRR@10. Its exact-identifier, conceptual, and operational Recall@30
are 1.0000, 0.8750, and 1.0000. The sole candidate miss is the stdout buffering
question. `isatty()` enters the candidates at rank 6; EOFError is rank 3, main
placement rank 2, and the main-guard question rank 13. Therefore optional local
multi-query expansion was not triggered: Development Recall@30 is 25/26 and the
principal missing `isatty()` case is resolved. No unbounded rewrite search was
performed.

### Frozen Holdout and Gate A

The selected configuration was opened on Holdout once after freezing. The
recommended-v1 comparison was evaluated only to add the same candidate metrics;
it was not used to change field/model parameters.

| Holdout candidate | Recall@10 | Recall@20 | Recall@30 | Hit@5 | MRR@10 | Candidate mean | Total retrieval mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recommended-v1 | 0.9444 | 0.9444 | 0.9444 | 0.8333 | 0.8519 | 0.0221 s | 0.1456 s |
| **Equal field + BGE-M3** | **0.8333** | **0.9444** | **0.9444** | **0.8889** | **0.8611** | **0.0385 s** | **0.1537 s** |

Recall@30 is unchanged at 17/18 while Hit@5 and MRR improve. Exact-identifier
and conceptual Recall@30 are both 1.0000; operational is 0.8333. All specified
Holdout hard cases are in the winner's candidates: `aclosing()` and
`dataclasses.replace()` rank 1, descriptor and Protocol rank 1, TaskGroup rank 5,
argparse mutually exclusive groups rank 13, `re.fullmatch()` rank 12, and
`ensure_ascii` rank 5. recommended-v1 missed argparse at 30; the selected method
instead misses the non-hard-case unittest subtest question. Candidate generation
is 16 ms slower on average, while full retrieval including reranking is 8 ms
slower, which remains acceptable for interactive CLI use.

Gate A passes: Holdout Recall@30 is not lower, Hit@5 and MRR improve, the named
hard-case candidate set does not regress, the artifact is pinned and reproducible,
and all citation/source metadata boundaries remain unchanged. This freezes the
Phase A retrieval winner for generator comparison; it does not change any CLI
default or recommended-v1.

### Phase A detailed artifact SHA-256

| Artifact | SHA-256 |
| --- | --- |
| `recommended_v1_development.json` | `61cf68cded0b5daab40da560da1e4036140af244adaa04b2caf2b9af97c78521` |
| `recommended_v1_holdout.json` | `35a590283fbf8604eb3b245911dc0712431532ffc6729c4b485f20ee14b5ae2f` |
| `field_equal_development.json` | `975288991f19355d67d5e98411661a4958eef7f8f9bcada1b7ea5ae8f50c45da` |
| `field_identifier-priority_development.json` | `130f8a5836495f45336a1d51c334cc215680272e2cc5e5f93c6ae20e5a99d89c` |
| `field_title-priority_development.json` | `ebaf775c31b6aa3e234a09afe22af936d8a892fb988a51d2b74c8f56e93f841e` |
| `field_identifier-title-priority_development.json` | `dfa4fe11132bcdc2444be361b1929c6b1aacffb5294420e8dfc3eb138312a0fd` |
| `bge-m3_embedding_development.json` | `028a8a42adbf2ac7f0854a5525b4aa2e7c53666de18c8e0f74bc7fb3a48d6ee8` |
| `bge-m3_field-embedding_development.json` | `3da1b77545a953a47e82958b6bddb78f9a47b59547846b325ff624b81adb8383` |
| `bge-m3_field-embedding_holdout.json` | `b6a3c469d674f393eb598c8879659d5a5cef7fdd741fc341c65fec51f541c7be` |
| `multilingual-e5-base_embedding_development.json` | `f79b6135402ad2252c136383135a63de4aefc8d863ec012b00c1e46a9b805e9a` |
| `multilingual-e5-base_field-embedding_development.json` | `51525dfbcd2aadbdbec053aa070c143b75186825114ce277011d48deff394e1d` |
| BGE-M3 manifest | `bf215417a87587eaefba4ad3dad98a4fcbc895bfeb479430647fe285158dd8ac` |
| E5-base manifest | `dbaef46c3d9a4e94f1cca5ec323666132f3b22f8ca622f4431d8cf6c06eb9ecc` |

Detailed JSON, sidecars, alternative indexes, model caches, and performance logs
remain outside Git below the data root.

## Phase B: Local Generator Tournament

Evaluation harness commits are
`733f186a8b929c54bca0dbb0ccfb9383dbbbc1d2` and
`27f46dc6bbe16f8911a22e33d1bd688ba4ec398a`. The Phase A winner retrieved and
reranked each question once. The resulting five-chunk tuple, rerank scores,
original ranks, citation metadata, sanitized prompt-visible context, and a
canonical tuple hash were saved before any generator comparison. The artifact
has 22 protected Answerability/RAG records; the extended artifact has those
same records plus four previously unevaluated Development hard cases. All
generators received the same saved tuple. Retrieval was not re-executed within
a generator run, and initial/retry prompts used that same tuple.

### Bounded model selection

The search stopped after the two predeclared additional candidates. All three
models load with Transformers, `trust_remote_code=False`, unquantized BF16, and
a pinned revision. No quantized fallback was used.

| Candidate | Pinned revision | Official model-card basis | Decoding | License |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` baseline | `cdbee75f17c01a7cc42f958dc650907174af0554` | Existing fixed multilingual recommended-v1 baseline | greedy | Apache-2.0 |
| **`Qwen/Qwen3-8B`** | `b968826d9c46dd6066d109eabc6255188de91218` | [100+ languages and multilingual instruction following](https://huggingface.co/Qwen/Qwen3-8B) | official non-thinking template; temperature 0.7, top-p 0.8, top-k 20, seed 20260731 | Apache-2.0 |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | [Japanese among 29+ languages and structured-output improvements](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05, seed 20260731 | Apache-2.0 |

Thinking/reasoning text was not requested, displayed, or stored. The Qwen3-8B
chat template used the official `enable_thinking=False` switch. Each process
loaded one tokenizer/model pair and reused it for every question. The first
candidate download is included in the initially observed load time; cached
loads are listed separately below. Qwen2.5's first Network Volume download hit
the cache quota, so the exact pinned revision was downloaded to an ephemeral
local cache without deleting or changing any experiment artifact.

### Contract and answerability results

| Dataset / Generator | Valid answer | Valid abstain | Contract failure | False answer | False abstain | Correct source cited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Answerability / 4B | 7 | 5 | 0 | 1 | 0 | 6/6 answerable |
| Answerability / **Qwen3-8B** | 6 | 6 | 0 | **0** | 0 | 6/6 answerable |
| Answerability / Qwen2.5-7B | 6 | 4 | 2 | 0 | 0 | 6/6 answerable |
| RAG quality / 4B | 8 | 1 | 1 | 0 | 2 | 8/10 |
| RAG quality / **Qwen3-8B** | **9** | 0 | 1 | 0 | **1** | 8/10 |
| RAG quality / Qwen2.5-7B | 8 | 2 | 0 | 0 | 2 | 7/10 |
| Added hard cases / 4B | 4 | 0 | 0 | 0 | 0 | 4/4 |
| Added hard cases / **Qwen3-8B** | 4 | 0 | 0 | 0 | 0 | 3/4 |
| Added hard cases / Qwen2.5-7B | 3 | 1 | 0 | 0 | 1 | 2/4 |

The Phase A retrieval winner exposed a useful generator distinction on the
unanswerable organization-policy question: the 4B model inferred a nonexistent
company convention from generic private-name examples, whereas both larger
models abstained. Qwen3-8B also changed one RAG false abstention into an answer.
The remaining `ensure_ascii=False` case still failed the v1 JSON contract for
4B and Qwen3-8B; Qwen2.5 abstained. This directly motivates Phase C.

Codex performed a source-grounded review of every answer against the frozen
source and required facts; the formal OpenAI Judge helper and API were not
invoked.
Conservative required-fact coverage over the ten RAG cases
plus four added hard cases is 22/35 for 4B, **25/35 for Qwen3-8B**, and 22/35
for Qwen2.5. Major semantic errors are 3, **2**, and 2 respectively. Remaining
Qwen3-8B defects are material: its argparse code is correct but the prose
incorrectly says `required=True` belongs on each option, and it misleadingly
describes `main()` itself as being under the guard. All models omit the
`read()`/`readline()` EOF distinction; all remain incomplete on descriptor
precedence, and Qwen3-8B still omits `await` and early-exit cleanup for
`aclosing()`. These negative results are retained in the external review.

### L4 performance

| Generator | Answerability generation mean | RAG generation mean | Peak allocated | Peak reserved | Peak CPU RSS | Cached load observation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | 2.397 s | 4.531 s | 8.79 GB | 9.02 GB | 5.02 GB | 1.40 s |
| **Qwen3-8B** | 2.805 s | 6.926 s | 17.15 GB | 17.48 GB | 8.01 GB | 1.39 to 5.25 s |
| Qwen2.5-7B | 3.708 s | 6.137 s | 15.89 GB | 16.33 GB | 10.21 GB | 0.92 to 0.95 s |

Qwen3-8B and Qwen2.5 both fit within the tested 23,034 MiB L4 without
quantization. These timings and memory values are specific to this L4.

### Gate B and detailed artifacts

Qwen3-8B passes Gate B and is frozen as the Generator winner: false answers are
zero, total contract failures equal rather than exceed the 4B comparison
(one), major errors decline, false abstentions decline, and required-fact
coverage improves. Qwen2.5 is rejected because its two Answerability contract
failures increase the aggregate contract-failure count and its hard-case false
abstention reduces completeness. This selection does not change defaults or
recommended-v1.

| Artifact | SHA-256 |
| --- | --- |
| Frozen 22-question contexts | `ce72e8e98650df037594cc8f146a102034c8a31b6a8df9a86f2f51dfc59a797f` |
| Frozen contexts with 4 extra hard cases | `253f23aefe556c29f9240ed7bfde6333ec4c8488aabbe1bcd007314d966252d9` |
| Added hard-case definitions | `05d2d1de2b53555590259cb92903d2c2e7acf9f9a991c233346c0fbf9d40134e` |
| Source-grounded Codex review | `c788be21b0ac2a32a5f75b64ef6b849982d3b9de25ac52319252eb17af8d6e74` |
| 4B Answerability / RAG / hard | `4e49bf0915ebc7ea75b37f9fdac5355a60583e428f29a1d36c0885f5438a66f6` / `8d17bdbc5c895592bdd1bfe57b1a59dcfc19f2810ad5b9a9187ad7f66fb4f47d` / `47c818ff1eba35e3a0916e002235d403046c237bffc9c0c44f2136394f59d3b0` |
| Qwen3-8B Answerability / RAG / hard | `7d741525e1495c4d8dff0544c9b20d4ca80a602a3d9262ef0468b0b6149b67fd` / `7e5d39efa6b17dab3a224d6e74928063fc57738f8918bd61210a5666f5bc82c5` / `dd15d5fcc40fd0be8a0458581df3bb5c0dc56485b63bb9646c74043fce0b2d87` |
| Qwen2.5 Answerability / RAG / hard | `4aa04d534a02c3cb7db7f90ca1594f6d545a589ce26c34b527cfbacb92fc1dc5` / `64976f0ebb952e9dd849e685ca6f4d95fb403caed1a8e5a94f3439e4b7941cb8` / `039ce98ca1e76dcf8d2fdd06cc41e8c1dab1bbde4f875d43dbb44935da83b54a` |

All detailed JSON, fixed contexts, review notes, and model caches remain outside
Git. Every artifact states that OpenAI API use is false and contains no secret.

## Phase C: Output Constraint Tournament

Implementation commit:
`c34721c83676b7ca47913cc8ad80fe9a667405b7`.

Candidate C1 (`two-stage-answerability-v1`) uses a model-independent token trie
and Transformers `prefix_allowed_tokens_fn` so Stage 1 can generate exactly
`answer` or `abstain`, with greedy decoding and no free-form JSON. `abstain`
returns immediately with no answer text or sources. `answer` invokes Stage 2,
which emits normal citation-bearing prose rather than a JSON string, rejects
URLs/Markdown links/invalid citations through the existing finalizer, and has
one retry that never receives the invalid first output. All stages reuse one
Qwen3-8B load and the same frozen five-chunk tuple. The exact serialized prompt
for each possible stage is checked against the token budget before generation.

C2 was skipped. No already-available lightweight integration could guarantee a
strict schema in which code fences and extra text were impossible; adding a new
constrained-generation dependency was therefore not justified. No project
dependency or lock file changed.

| Dataset / Qwen3-8B constraint | Valid answer | Valid abstain | Contract failure | False answer | False abstain | Correct source cited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Answerability / v1 JSON | 6 | 6 | 0 | 0 | 0 | 6/6 |
| Answerability / **two-stage** | 6 | 6 | 0 | 0 | 0 | 6/6 |
| RAG quality / v1 JSON | 9 | 0 | 1 | 0 | 1 | 8/10 |
| RAG quality / **two-stage** | **10** | 0 | **0** | 0 | **0** | **9/10** |
| Added hard / v1 JSON | 4 | 0 | 0 | 0 | 0 | 3/4 |
| Added hard / **two-stage** | 4 | 0 | 0 | 0 | 0 | 3/4 |

The two-stage candidate fixes the target failure: `ensure_ascii=False` changes
from final JSON contract failure to a valid, correct cited answer. Abstentions
leak neither answer text nor sources. All 26 Stage 1 choices are exact and valid.
The price is additional calls: 19 for 12 Answerability questions, 20 for ten RAG
questions, and eight for four hard cases, versus one call per direct success and
one additional call only on direct retry.

Semantic quality does not follow contract quality automatically. Conservative
required-fact coverage rises only from 25/35 to 26/35. Two-stage improves
`dataclasses.replace`, `ensure_ascii`, and one EOFError condition, but still
misses required facts for `aclosing()`, descriptor precedence, TaskGroup, and
EOF behavior. More importantly, its argparse answer invents per-option
`required=True` and `conflict=True`; the correct argparse source was absent from
the selected Top-5. Main placement also remains misleading. Major semantic
errors remain two, and the argparse error is more severe than the direct
candidate's mixed prose/code answer. These defects remain explicit inputs to
the final winner selection.

| Dataset | Mean generation time | Mean total time | Calls | Peak CUDA allocated / reserved | Peak CPU RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Answerability | 3.219 s | 3.256 s | 19 | 17.04 / 17.41 GB | 5.04 GB |
| RAG quality | 6.195 s | 6.231 s | 20 | 17.11 / 17.40 GB | 5.05 GB |
| Added hard cases | 7.850 s | 7.887 s | 8 | 16.97 / 17.25 GB | 5.05 GB |

Gate C passes on the predefined criteria: zero false answers, zero abstain
leakage, fewer contract failures, no false-abstention increase, maintained or
better citation success, a fixed `ensure_ascii=False` case, model-independent
constraints, and reusable model state. Gate passage is not a production or
recommended-profile decision.

| Phase C artifact | SHA-256 |
| --- | --- |
| `two-stage_answerability.json` | `fa0b9187475bd8d4ce1b7c42ca6761d3b2779f5ca585cd8ae75767f3fc292e20` |
| `two-stage_rag-quality.json` | `043d9cd295da8a70265b5816a9fa65253d5225d3524d98cb613b8610f358eecf` |
| `two-stage_hard-cases.json` | `6f91b03270cf9e04de4d57d90e217030f999b48aa14f384e700cf3550a90ea99` |
| Source-grounded Codex review | `dbfb4da968676ab1c8cb8b31af1032cf930fbc736f721671f5e1e37d991022ea` |

## Phase D: Evidence-first Generation

Implementation commit:
`8d0eb564a02dea0b4150603f9a2cd7b04c9bcf3b`.

Phase D was triggered because direct Qwen3-8B remained materially incomplete
on at least `aclosing()` and descriptor precedence. `evidence-first-v1` asks for
at most three short facts, each at most 240 characters, and accepts a fact only
when it is an exact whitespace-normalized substring of its declared sanitized
source. Invalid labels, duplicate sources, unsupported spans, URLs, schema
drift, and excess length/count fail closed. The answer prompt receives only the
validated spans and source labels; evidence JSON is hidden from the user. Both
extraction and answer generation have one retry that does not receive rejected
text, and every question continues after an isolated failure.

| Dataset | Valid answer | Valid abstain | Contract failure | False answer | False abstain | Correct source cited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Answerability | 4 | 3 | 5 | **2/6 unanswerable** | 4/6 answerable | 2/6 answerable |
| RAG quality | 3 | 0 | 7 | 0 | 7/10 | 2/10 |
| Added hard cases | 1 | 0 | 3 | 0 | 3/4 | 0/4 |

This is a clear negative result. Extractive matching prevents invented evidence
text but cannot establish relevance to the full question. Qwen3-8B selected a
generic `visit_` recommendation and invented an organization policy, then used
a shell `alias` passage to invent that a local `python` alias resolves to
`python3`. False answers therefore rise from zero to two. Exact spans also prove
too brittle: model paraphrases or formatting changes fail evidence validation
twice in five Answerability cases, seven RAG cases, and three hard cases.
Conservative RAG-plus-hard coverage collapses to 5/35. The one EOF hard answer
focuses on unrelated connection/console behavior, while even the accepted
`dataclasses.replace` answer drops an extracted `__init__()` fact.

| Dataset | Mean generation time | Calls | Peak CUDA allocated / reserved | Peak CPU RSS |
| --- | ---: | ---: | ---: | ---: |
| Answerability | 9.369 s | 22 | 17.07 / 17.63 GB | 5.04 GB |
| RAG quality | 15.020 s | 20 | 17.12 / 17.49 GB | 5.05 GB |
| Added hard cases | 14.067 s | 8 | 16.99 / 17.52 GB | 5.05 GB |

Gate D fails: false answers and contract failures increase, completeness and
correct citations fall sharply, and latency rises. Evidence-first is frozen as
rejected and is excluded from Phase E rather than tuned after seeing results.

| Phase D artifact | SHA-256 |
| --- | --- |
| `evidence-first_answerability.json` | `90cf84ef0462662acaaf3074a7453cbb83e08a56ad6dd464fe65bd00b9fc090a` |
| `evidence-first_rag-quality.json` | `b0ab3ae067b450ad1210764e95ceaae3b8a12bab94a5a1b237761627ca5e147e` |
| `evidence-first_hard-cases.json` | `3e26ff58fb9dfcd9afa656dc97e4b6a30fe60c0be0652a72f1325faf26691416` |
| Source-grounded Codex review | `0ae158898b6272c081e6b368a6e9321c778d2dfdd6e08cd32eca89899304f30d` |

## Phase E: Frozen Combination Tournament

Phase E performed no search. It combined only the settings already frozen in
Phases A through C. Evidence-first was excluded because Gate D failed. The
recommended-v1 contexts were frozen independently and verified byte-for-byte
against its earlier Answerability and RAG detailed results; this prevented a
retrieval rerun from contaminating the generator comparison.

| ID | Retrieval / generator / output | Answerability A / abstain / fail | False answer | RAG A / abstain / fail | RAG correct cited | Hard A / abstain / fail | Hard correct cited |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | recommended-v1 / 4B / v1 | 6 / 6 / 0 | 0 | 8 / 1 / 1 | 8/10 | 4 / 0 / 0 | 2/4 |
| B | Phase A / 4B / v1 | 7 / 5 / 0 | **1** | 8 / 1 / 1 | 8/10 | 4 / 0 / 0 | 4/4 |
| C | recommended-v1 / 8B / v1 | 6 / 6 / 0 | 0 | 9 / 0 / 1 | 8/10 | 3 / 1 / 0 | 2/4 |
| D | recommended-v1 / 4B / two-stage | 6 / 6 / 0 | 0 | 10 / 0 / 0 | 8/10 | 3 / 1 / 0 | 2/4 |
| E | Phase A / 8B / v1 | 6 / 6 / 0 | 0 | 9 / 0 / 1 | 8/10 | 4 / 0 / 0 | 3/4 |
| F | Phase A / 8B / two-stage | 6 / 6 / 0 | 0 | **10 / 0 / 0** | **9/10** | 4 / 0 / 0 | 3/4 |

Candidate B is ineligible because the 4B model answers the unanswerable
organization-policy question. C and D demonstrate that a stronger generator
or stricter output structure cannot repair missing candidates: the
recommended-v1 Top-5 lacks the expected source for `isatty()`, argparse, and
`ensure_ascii`. D structurally succeeds but generates unsupported claims that
the two argparse options are individually required and that a nonexistent
`--no-ensure-ascii` option should be used. It also gives an unrelated
connection-specific EOF explanation and retains the incorrect `main`
placement. It is excluded from the final tournament.

E and F are the only new finalists. E is the strongest direct-output
candidate, with 25/35 conservative required-fact coverage and two major
semantic errors. F fixes `ensure_ascii=False`, has no contract failure, cites
the correct source for 9/10 RAG questions, and reaches 26/35 facts, but it also
has two major errors: unsupported argparse flags and misleading `main`
placement. The untouched evaluation, not these already-observed questions,
will decide between A, E, and F. Their complete settings are frozen before the
new questions are created; the new results may select a profile but cannot
change a parameter or add a feature.

| Phase E artifact | SHA-256 |
| --- | --- |
| recommended-v1 frozen 26-question contexts | `712b00fddd2d671dade0d42f064f884b1204e36a9ca6af46c748c5e62a019329` |
| A hard cases | `f52c855eca43cb1bd0fa4425f25be4ebb31fdd0dc1c601bba1668f9bf2d93de4` |
| C Answerability / RAG / hard | `81995081371bf9d6744585285893cde7533adc55d18b44688e698b37cc5a285f` / `451b4c5a17634d5a96274b9ced78fc078735e1e9924771a5558f95c4b816571c` / `151f42e192f13139b045632c103a14348c265ffba8020214e6a134b0af4bc349` |
| D Answerability / RAG / hard | `db258721b8fb8ac06347d8c8571455a09031619bd15e5ca658794d22d1daab98` / `6d4918787762d999d5a60cf0bf562561b86c1ab7b535d1060499f8b39c8a33af` / `194293ab4daa9af32082c4c84d82e5483f76a90ca74a2b915823a540ee9a4e61` |
| Source-grounded combination review | `582911dd3d130622edf547c8fe39990e556f6a590769d43fcc8cd6379289ff20` |

## Remaining phase

The frozen final candidates are recommended-v1 (A), E, and F. The untouched
question definitions are committed before they are evaluated once.
`recommended` still has exactly the recommended-v1 meaning pending that result.

## Final Untouched set definition

The set was created only after Phase E and its frozen A/E/F settings were
committed. Selection used no random sampling: protected questions and the 13
known hard-case APIs were first excluded; corpus rows were then checked in
processed-JSONL order against a fixed topic sequence. RAG order is four exact
identifier topics (`asyncio-queue`, `statistics`, `enum`, `mimetypes`), four
conceptual topics (`copy`, `weakref`, `decimal`, `ssl`), then four operational
topics (`zipfile`, `sqlite3`, `venv`, `importlib-resources`). Each selected
answer and required fact was verified in the saved SearchChunk text and each
uses a distinct source URL/anchor. No question is a paraphrase of an earlier
Development, Holdout, Answerability, RAG-quality, or hard-case question.

The Answerability set uses four independently verified Python 3.13 topics
(`os.path`, `base64`, `xml.etree.ElementTree`, and `dbm.sqlite3`) and four
realistic later-version questions. The latter APIs (`annotationlib`,
`compression.zstd`, `uuid.uuid7`, and `InterpreterPoolExecutor`) were confirmed
absent from the byte-verified Python 3.13 corpus and include the reason for
abstention. They are not meaningless strings or environment secrets.

| Frozen question file | Count | SHA-256 |
| --- | ---: | --- |
| `final_untouched_rag_questions.jsonl` | 12 (4 exact / 4 conceptual / 4 operational) | `098a020aa24e46c65faa4e3d4bd84cfd95b0878b89cbbf189376733d4153681e` |
| `final_untouched_answerability_questions.jsonl` | 8 (4 answerable / 4 unanswerable) | `098983c7169a009d2b5633759ee31cf2c7feeab7b03f74b0e9b8fe8cfb38610a` |

After this definition is committed, recommended-v1 and finalists E/F are run
once. The outputs cannot be used to retune retrieval, generation, prompts, or
constraints; only the winner profile may be selected.

## Final Untouched results and winner

Settings were frozen in commit
`b2bbc8824937f55367d7adcb34ed582a58e9fb89`; the new questions were then
committed as `0881a7373688973b8c055b3a7c9495c35bcf0e9c` before any final run.
Recommended-v1 and finalists E/F were each evaluated once. No final result was
used to modify a question, parameter, prompt, retrieval component, or output
contract.

| Final metric | A: recommended-v1 | E: Phase A + 8B + v1 | F: Phase A + 8B + two-stage |
| --- | ---: | ---: | ---: |
| Unanswerable false answer | **1/4** | **1/4** | **1/4** |
| Answerable false abstention | 1/4 | **0/4** | **0/4** |
| Valid-abstain answer/source leakage | 0/3 | 0/3 | 0/3 |
| Contract failure (all 20) | 0 | 0 | 0 |
| RAG valid answer | 9/12 | **12/12** | 11/12 |
| RAG correct source Top-5 | 10/12 | **12/12** | **12/12** |
| RAG correct source cited | 8/12 | **11/12** | 10/12 |
| Conservative RAG required facts | 19/29 | **24/29** | 23/29 |
| RAG major semantic errors | 2 | 2 | 2 |
| RAG unsupported material claims | 2 | 2 | 2 |

The primary safety target was not achieved. All three answer the out-of-corpus
`InterpreterPoolExecutor` question by borrowing ThreadPoolExecutor or
ProcessPoolExecutor's `initializer`/`initargs` behavior and display that
irrelevant source. The other three unanswerable cases abstain with no text or
source leakage. This result remains untouched and must not be hidden by the
winner decision.

A also abstains on answerable `os.path.isreserved` and on three RAG questions.
Its candidate set misses the expected zipfile and sqlite sources. E includes
all twelve expected RAG sources in Top-5 and answers all twelve. F uses the same
contexts but abstains on `statistics.kde`; its two-stage structure provides no
final contract benefit because A/E also have zero failures. Across RAG, all
three have misleading wording that a live weakref finalizer is repeatedly
called while `atexit` remains true, and internally contradictory decimal
wording before eventually stating that equality does not raise. E nevertheless
has the highest conservative completeness without an added major error.

The first three winner priorities are tied: one false answer, zero leakage on
valid abstentions, and zero contract failures. E then wins by the predefined
ordering: fewer false abstentions, more correct citations, complete final
candidate Top-5, better completeness, and the already-frozen Holdout Hit@5/MRR
improvement. F loses on final false abstention and citation completeness while
making more generation calls. The selected profile is therefore:

```text
recommended-v2 = technical-field-retrieval-v1
               + BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181
               + equal field RRF Top-30
               + mMARCO MiniLM reranker Top-5
               + Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218
               + official non-thinking sampling
               + answer-or-abstain-v1
```

`recommended` now aliases `recommended-v2`. `recommended-v1` remains unchanged
as a rollback profile, and the profile-less Dense/legacy defaults remain
unchanged. No two-stage or evidence-first component is in v2.

### Final L4 reference performance

These generator timings use frozen contexts and therefore intentionally omit
candidate retrieval and reranking time. The actual recommended-v2 CLI smoke
subsequently measured retrieval at 0.341 s for one cached-model query.

| Candidate | Answerability generation mean | RAG generation mean | RAG total mean | Peak CUDA allocated / reserved | Peak CPU RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1.935 s | 4.151 s | 4.162 s | 8.77 / 9.10 GB | 5.03 GB |
| E | 3.472 s | 7.200 s | 7.211 s | 17.08 / 17.56 GB | 5.05 GB |
| F | 3.155 s | 5.869 s | 5.903 s | 17.04 / 17.47 GB | 5.05 GB |

The 8B winner fits unquantized BF16 on the 23,034 MiB L4. Its latency and VRAM
cost are materially higher than v1 and are part of the documented trade-off.

### Final artifacts

| Artifact | SHA-256 |
| --- | --- |
| recommended-v1 final frozen contexts | `671c9f33ea4c86ea52bdf3979091cc9ab40c0a0bee482ecd4241199d710ee507` |
| Phase A winner final frozen contexts | `fe0039d87381bfeebfc13db14efb4d5331fcc074ca5ef6ad821db9d9128305e5` |
| A Answerability / RAG | `9e14aecc019184f3d226c9694af38aaeedc43c1ed3c993c4e2e305071aa932e9` / `7f267bcacdd857ccb8d86868c0661ce11b8e717804a106871db71c7b809d1268` |
| E Answerability / RAG | `aa58c25ca39b0000c142a344dd428de41165727ee97b626d91a410b06474c4a8` / `1cf27162bbcb2024ca45b3e660a678876066f0f9b58df41b5c967ee382d67fdc` |
| F Answerability / RAG | `230c5993303ef449e0ceb50fee04ab31c36c424d065ec6d96a3906a6b7a30217` / `8d887ced61ea4641b2049429ea6c9474392d2c812b91cabf382a792850f4e7a2` |
| Final source-grounded review | `f44b532105b91ee5e114475fe73ef0ce8d582a0875779ba1050bac8902f4c916` |

All detailed outputs, contexts, reviews, indexes, and model caches remain
outside Git. Every runner records `openai_api_used: false` and
`contains_secrets: false`; OpenAI API and OpenAI Judge were not used.

## Final disposition

Python-specific quality work is frozen at recommended-v2 with the explicit
`InterpreterPoolExecutor` false-answer limitation. The symbol/field retrieval,
artifact validation, model pinning, prompt/citation safety boundary, and
profile wiring are reusable for general HTML ingestion. Generalization still
requires a new corpus-specific blind evaluation; this Python result must not be
assumed to transfer automatically.
