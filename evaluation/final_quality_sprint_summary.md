# Final quality sprint evaluation

## Scope and provenance

- Git branch: `feat/final-quality-sprint`
- Start commit: `b767b76ed0e13391bee84a7359878a33c9e454ff`
- Original start: `2026-07-31T18:51:13Z`
- Resume after raw restore: `2026-07-31T19:20:26Z`
- Questions and baseline artifacts were verified against their existing SHA-256.
- The restored raw HTML produced 8,677 chunks byte-identical to the protected
  baseline processed JSONL.
- OpenAI API and OpenAI Judge were not used.

This is a historical evaluation record. At this checkpoint, `recommended`
identified the configuration now retained as `recommended-v1`; the current
checkout instead aliases `recommended` to `recommended-v2`. Historical commands
below assumed a prepared data root supplied by the environment.

Experiments were run on an NVIDIA L4 (23,034 MiB), driver `580.126.20`, PyTorch
`2.11.0+cu128`, CUDA runtime 12.8. Results from a different GPU are not used for
latency or resource comparisons.

## Phase A: Local Reranker

Revision: `local-cross-encoder-rerank-v1`.

The reranker receives pairs of the question and a document string containing only
`page_title`, `section_title`, and chunk `text`. It does not generate candidates or
change `SearchChunk` metadata. Scoring is batched, each model is loaded once per
run, score ties retain the original rank, and reranking diagnostics remain outside
the Generator prompt. Citation URLs continue to come only from the selected
chunk metadata.

### Bounded model candidates

| Key | Model / pinned revision | Model-card evidence | License |
| --- | --- | --- | --- |
| `bge-m3` | `BAAI/bge-reranker-v2-m3` / `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | [Multilingual retrieval reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3), XLM-R sequence classifier, about 0.6B parameters | Apache-2.0 |
| `mmarco-minilm` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` / `1427fd652930e4ba29e8149678df786c240d8825` | [Japanese is explicit in the language metadata](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1), Sentence Transformers CrossEncoder, about 0.1B parameters | Apache-2.0 |

Both load through the existing Sentence Transformers dependency with
`trust_remote_code=False`. No third model was considered.

### Phase A development selection

Only the 26-question Development set was used for model and parameter selection.
The order was Hit@5, MRR@10, then the smaller/faster configuration. Hybrid uses
the existing fixed Japanese bigram RRF configuration.

| Model | Candidates | k | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Rerank mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE M3 | Dense | 20 | 0.5769 | 0.8077 | 0.8077 | 0.8077 | 0.6731 | 0.7129 s |
| BGE M3 | Dense | 30 | 0.5769 | 0.7692 | 0.8077 | 0.8077 | 0.6679 | 1.0834 s |
| BGE M3 | Hybrid | 20 | 0.5769 | 0.8077 | 0.8462 | 0.8846 | 0.6927 | 0.7386 s |
| BGE M3 | Hybrid | 30 | 0.5000 | 0.8077 | 0.8462 | 0.8846 | 0.6554 | 1.1009 s |
| mMARCO MiniLM | Dense | 20 | 0.5385 | 0.7692 | 0.8077 | 0.8077 | 0.6487 | 0.0840 s |
| mMARCO MiniLM | Dense | 30 | 0.5000 | 0.7692 | 0.7692 | 0.8077 | 0.6273 | 0.1199 s |
| mMARCO MiniLM | Hybrid | 20 | 0.5385 | 0.8462 | 0.8462 | 0.8846 | 0.6722 | 0.0810 s |
| **mMARCO MiniLM** | **Hybrid** | **30** | **0.5000** | **0.8077** | **0.8846** | **0.8846** | **0.6667** | **0.1172 s** |

Development baselines were Dense Hit@5 0.6923 / MRR@10 0.4972 and Hybrid
Hit@5 0.6923 / MRR@10 0.6389. The selected setting is therefore mMARCO
MiniLM, Hybrid candidates, candidate k 30, batch size 16, maximum pair length
512. BGE Hybrid k 20 had higher MRR, but mMARCO Hybrid k 30 had the highest
Hit@5 and was about nine times faster in this bounded run.

### Gate A evaluation

The selected setting was run on Holdout exactly once.

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | 0.5556 | 0.6667 | 0.7778 | 0.9444 | 0.6492 |
| Rerank Hybrid | 0.8333 | 0.8333 | 0.8333 | 0.9444 | 0.8519 |

| Answerability metric | Hybrid baseline | Rerank Hybrid |
| --- | ---: | ---: |
| Answerable answer | 6/6 | 6/6 |
| False abstention | 0/6 | 0/6 |
| Unanswerable abstention | 6/6 | 6/6 |
| False answer | 0 | 0 |
| Contract failure | 0 | 0 |
| Correct source Top-5 / cited | 6/6 / 6/6 | 6/6 / 6/6 |

| RAG quality metric | Hybrid baseline | Rerank Hybrid |
| --- | ---: | ---: |
| Valid answer | 7/10 | 8/10 |
| Valid abstain | 2/10 | 1/10 |
| Contract failure | 1/10 | 1/10 |
| Correct source Top-5 | 7/10 | 8/10 |
| Correct source cited | 7/10 | 8/10 |

Answerability mean prompt size was 2,141.6 tokens, mean retrieval including
reranking was 0.1612 seconds, and mean total time was 2.1571 seconds. RAG mean
prompt size was 2,048.8 tokens, mean retrieval was 0.1654 seconds, and mean total
time was 4.8112 seconds. Combined Qwen and reranker peak allocated GPU memory was
9.70 GB or less and peak process RSS was 5.84 GB or less.

Gate A passes: Holdout Hit@5 and MRR improved, correct-source metrics did not
decrease, false answer stayed zero, contract failure did not increase relative to
the selected Hybrid baseline, and RAG valid answers increased.

### Known hard cases after reranking

- `isatty()` remains outside Top-10; this is unresolved but unchanged from Hybrid.
- `EOFError` improved from rank 7 to 2.
- the `__main__` guard improved from rank 8 to 2; main placement improved from
  rank 9 to rank 1.
- `re.fullmatch()` is now answered correctly from the correct source.
- `asyncio.TaskGroup` changed from abstention to a materially correct answer that
  covers cancellation, waiting, and ExceptionGroup handling.
- descriptor precedence remains correct.
- `contextlib.aclosing()` remains incomplete because the answer omits awaiting
  `aclose()` and deterministic cleanup on early exit.
- argparse mutually exclusive groups remain a false abstention.
- `ensure_ascii=False` remains the same final contract failure as Hybrid baseline.

No new major semantic regression was found in the fixed hard-case review.

### Phase A detailed artifact SHA-256

| Artifact | SHA-256 |
| --- | --- |
| `bge-m3_dense_k20_development.json` | `19a4f9c757cfe74c6045ea2fa5b92ad89242e99ab7599d02a612569941d4c488` |
| `bge-m3_dense_k30_development.json` | `117654d61dcca4da9c9ee99a815175cbd604bf3163238fa0749ec583834705ab` |
| `bge-m3_hybrid_k20_development.json` | `f093af53ee487171368e4d48c3a6231603d999af5fde744d613f61874aeaa4e6` |
| `bge-m3_hybrid_k30_development.json` | `0ddeb41c1ff4d5d1efe39f7e0857b14ea2a6190bd2c2cb1e805acf50d7b34467` |
| `mmarco-minilm_dense_k20_development.json` | `748fd9acb03f356da5b4514c910eb4136ca106de4aca1ae3b1a905a525947ae4` |
| `mmarco-minilm_dense_k30_development.json` | `813ff807ee2e7b2c25728391e2928997384aebcf5a738916ce471d862501f6e7` |
| `mmarco-minilm_hybrid_k20_development.json` | `299b954e4003cb71a5496d9a3e5ce4bc6913b80038220c3f4c349bb4c53078b9` |
| `mmarco-minilm_hybrid_k30_development.json` | `1e7a1bd4baf76e8fd2d749f79e129257692f97d7fb9d059586b65f789b21d153` |
| `mmarco-minilm_hybrid_k30_holdout.json` | `9fc6b3da85b08a1d40e7cf4633a2f044705a889df2f15654f63f372789311dc3` |
| `answerability_rerank_hybrid.json` | `2b550274604c4893e32f9246408283c7a45a617884b79c5a20cfec49abce6783` |
| `rag_quality_rerank_hybrid.json` | `7b2a78bad27c1245bcc7044fde740846cfa203b1ff956bcb724a6b44737fd2e6` |

Detailed JSON and model caches remain outside Git under the data root.

## Phase B: Full DocumentSection Parent

Revision: `section-parent-v1`.

The restored raw HTML was parsed with the existing production parser into the
complete heading-scoped `DocumentSection` objects, rather than reusing the
approximately 1,000-character baseline `SearchChunk` objects as parents. A stable
section ID is the SHA-256 of canonical non-text section metadata; the exact text
has an independent SHA-256. Persistence and child resolution reject duplicate
IDs, missing parents, source URL or metadata differences, and text hash or child
substring differences. Section and child artifacts are written atomically and do
not replace the protected baseline corpus or index.

The child is search-only. Generation receives a section with its original citation
metadata. If the entire section fits the exact initial and retry prompt budgets it
uses `full_section`; otherwise a deterministic `section_window` expands from the
matched child at paragraph boundaries. Internal IDs, offsets, and child text are
not added to the prompt, and citation URLs remain section metadata only. This is
therefore a token-aware section-parent design, not a claim that every complete
section is always sent unchanged.

### Section artifact

- HTML: 384 detected / 384 parsed; failures 0; missing categories 0.
- Sections: 2,766, matching the parser result; excluded short/empty sections 81;
  duplicate section IDs 0.
- `sections.jsonl` SHA-256:
  `25039412634189e9e62e6aa636357a426d15a2aca04193acdb1254b636139d7b`.
- The full two-index build took 152.96 seconds, peak process RSS was 2.67 GB,
  and peak allocated GPU memory was 622.9 MB.

| Child setting | Children | Mean / max per section | Index / metadata | Index build | Child JSONL SHA-256 | FAISS SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 300 / 75 | 28,234 | 10.208 / 337 | 43,367,469 / 33,244,730 bytes | 42.33 s | `b4be25cbf6a569eb644d5a6b6501d14c7dbab7fb13f8316bd40455c1f4dc4b34` | `8a281ba2ed9ade698f3c6ab490a354d6db99fe6441ce2524781f94a69c7dd211` |
| 400 / 100 | 21,343 | 7.716 / 255 | 32,782,893 / 27,427,403 bytes | 29.06 s | `9a2a33559ad6cf78e342f9c375b186b2a60ae5f30be75a598264ff0e863200e9` | `4e18d80b3566a2b23af00a09ff59d594e0c537304443d68ffb03d433564dc00a` |

### Phase B development selection

Only Development was used to select the bounded two-setting grid. Candidate k
30 produced at least 14.46 average unique section candidates in every run and
17.62 in the selected run, so the optional k 60 expansion was unnecessary.

| Child | Candidates | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Retrieval mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 / 75 | Dense | 0.4231 | 0.6154 | 0.6923 | 0.7692 | 0.5292 | 0.0295 s |
| 300 / 75 | Hybrid | 0.4615 | 0.5769 | 0.6923 | 0.7692 | 0.5526 | 0.0423 s |
| 400 / 100 | Dense | 0.4231 | 0.6538 | 0.6923 | 0.7308 | 0.5417 | 0.0064 s |
| **400 / 100** | **Hybrid** | **0.4615** | **0.6154** | **0.6923** | **0.8077** | **0.5769** | **0.0193 s** |

Hit@5 tied across the four section candidates, so 400/100 + Hybrid was frozen by
the highest MRR@10. It did not beat the Development Hybrid baseline MRR 0.6389;
that negative evidence was retained before opening Holdout.

### Gate B evaluation

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | 0.5556 | 0.6667 | 0.7778 | 0.9444 | 0.6492 |
| Section Parent Hybrid | 0.5000 | 0.8889 | 0.9444 | 0.9444 | 0.6620 |

| Answerability metric | Hybrid baseline | Section Parent Hybrid |
| --- | ---: | ---: |
| Answerable answer | 6/6 | 5/6 |
| False abstention | 0/6 | 1/6 |
| Unanswerable abstention | 6/6 | 6/6 |
| False answer | 0 | 0 |
| Contract failure | 0 | 0 |
| Correct source Top-5 / cited | 6/6 / 6/6 | 5/6 / 4/6 |

| RAG quality metric | Hybrid baseline | Section Parent Hybrid |
| --- | ---: | ---: |
| Valid answer | 7/10 | 8/10 |
| Valid abstain | 2/10 | 1/10 |
| Contract failure | 1/10 | 1/10 |
| Correct source Top-5 | 7/10 | 9/10 |
| Correct source cited | 7/10 | 7/10 |

Answerability used 43 full sections and 6 windows; RAG successful outcomes used
28 full sections and 10 windows. No question had zero fitting context, although
the budget excluded 11 and 7 lower-ranked sections respectively. Mean prompt size
was 6,649.0 tokens for Answerability and 6,825.2 for RAG, substantially larger
than chunk reranking. Mean retrieval was 0.0368 / 0.0361 seconds and mean total
time was 3.4731 / 6.6335 seconds. Peak process RSS was at most 5.93 GB, peak GPU
allocated 10.44 GB, and peak reserved 11.85 GB.

Gate B fails. Holdout retrieval and RAG valid-answer count improved, but
Development did not beat Hybrid, Answerability added a false abstention, and both
correct-source measures fell. The `re.fullmatch()` answer is the major semantic
regression: the correct operation is `re.fullmatch()`, but Qwen answered
`match()` while citing an irrelevant Match Objects section. `contextlib.aclosing()`
still omits awaiting `aclose()` and deterministic early-exit cleanup;
`argparse` still abstains; `ensure_ascii=False` remains a contract failure.
Section retrieval also lost `isatty()` and the main-guard case from Top-10;
`EOFError` and main placement were rank 6. Full-section parenting is therefore not
accepted alone, while the implementation remains a reusable research asset.

### Phase B detailed artifact SHA-256

| Artifact | SHA-256 |
| --- | --- |
| `section_manifest.json` | `8a6ea66ca96eb469fa05d65ca6299b84a0fe1093457a28cdf7da619c1e2a792d` |
| `section_300_75_dense_development.json` | `3ebbc614b823634ce494cce00506e898215a98df68ac8cb3b462a196ded6ff16` |
| `section_300_75_hybrid_development.json` | `34a1f93908a3fbe7c40e349658eadfbdc02a5e4e8ac03805f03c692639e001bd` |
| `section_400_100_dense_development.json` | `61880acf2eb3a1efca715f2fab1e533e5791d767c87609e7e5b06b6e8e84fafb` |
| `section_400_100_hybrid_development.json` | `d0b0e902986789c542858c227f4234074999e2db4090cfedce812f042181f6ba` |
| `section_400_100_hybrid_holdout.json` | `72476912c6ba6cc2affe89208eff50e78cf8640fbb7e923256025dc61aebfaf7` |
| `answerability_section_hybrid.json` | `8909959e6b22d8516a72961535ccdb03bb2ed5eb3819cc96013f7cbcb5a75054` |
| `rag_quality_section_hybrid.json` | `b0ca9182842b6b7025f0699a8465b4012eadf35c50e51da058b468a72ca51215` |

Detailed JSON, sections, child mappings, and indexes remain outside Git under the
data root.

## Phase C: Combined

Gate A passed, so the frozen Phase A and Phase B settings were combined without
another search: 400/100 section children, Hybrid child retrieval, child candidate
k 30, mMARCO MiniLM reranking, section candidate k 30, batch 16, maximum pair
length 512, and final Top-5 token-aware section contexts.

| Dataset | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Retrieval mean | Rerank mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 0.4615 | 0.7692 | 0.7692 | 0.8846 | 0.5936 | 0.1391 s | 0.1089 s |
| Holdout | 0.3889 | 0.7778 | 0.8889 | 1.0000 | 0.5880 | 0.1328 s | 0.0960 s |

| Quality metric | Hybrid baseline | Rerank only | Section only | Combined |
| --- | ---: | ---: | ---: | ---: |
| Answerability false answer | 0 | 0 | 0 | 0 |
| Answerability false abstention | 0/6 | 0/6 | 1/6 | 2/6 |
| Answerability contract failure | 0 | 0 | 0 | 0 |
| Answerability correct source cited | 6/6 | 6/6 | 4/6 | 4/6 |
| RAG valid answer | 7/10 | 8/10 | 8/10 | 7/10 |
| RAG contract failure | 1/10 | 1/10 | 1/10 | 1/10 |
| RAG correct source Top-5 | 7/10 | 8/10 | 9/10 | 8/10 |
| RAG correct source cited | 7/10 | 8/10 | 7/10 | 7/10 |
| Holdout Hit@5 | 0.7778 | 0.8333 | 0.9444 | 0.8889 |
| Holdout MRR@10 | 0.6492 | 0.8519 | 0.6620 | 0.5880 |

Combined Answerability had 4/6 answerable answers, 6/6 correct abstentions,
no false answer, and no contract failure. RAG had 7 valid answers, 2 valid
abstentions, and the same `ensure_ascii=False` contract failure. Answerability
used 50 full sections / 5 windows and RAG used 37 / 4; neither had a zero-context
question. Mean prompt tokens were 6,284.3 and 6,409.7, mean total times were
3.3606 and 6.1730 seconds, peak process RSS was 6.03 GB, peak allocated GPU
memory 10.91 GB, and peak reserved memory 11.69 GB.

The combined hard-case outcome is mixed: `EOFError` reached rank 1, the main
guard / placement reached 6 / 3, `re.fullmatch()`, `contextlib.aclosing()`,
descriptor precedence, and argparse were answered correctly. However `isatty()`
remained outside Top-10, dataclasses replacement and `asyncio.TaskGroup` became
false abstentions, and `ensure_ascii=False` still failed its contract. These are
material regressions against rerank only, so combination does not pass the gate.

| Phase C detailed artifact | SHA-256 |
| --- | --- |
| `section_400_100_hybrid_mmarco-minilm_development.json` | `00135620b45e9287845e280a94fd8a6070e05bd317bc4b5b65c868f0dac71879` |
| `section_400_100_hybrid_mmarco-minilm_holdout.json` | `8765230d3c50305ced32156b27f80d4ca97c9aec05c4460871b33684e966bd6c` |
| `answerability_section_rerank.json` | `8406967bc55c154a2b0136fa42f3ed0df2d5b2854b72949e02ab32840b05194e` |
| `rag_quality_section_rerank.json` | `9fb8c90267e50d5b64a494090bf898959c14737e65ab89b91a62cd10feb19d6c` |

## Tournament recommendation

The winner is **Rerank only**. It is the only new candidate that keeps
unanswerable false answers at zero, keeps abstention output/source leakage at
zero, does not increase contract failures versus its Hybrid input baseline,
keeps Answerability false abstention at zero, increases RAG valid answers and
correct-source citations, and materially improves both Holdout Hit@5 and MRR.
Section Parent has stronger Holdout Hit@5 but fails Development consistency and
Answerability; Combined inherits the section context cost and adds false
abstentions. No single retrieval metric overrides those contract and semantic
criteria.

At this checkpoint, the fixed command was the following; the environment
provided the prepared data root:

```bash
python -m python_doc_rag chat --profile recommended
```

On the current checkout, the equivalent configuration is selected explicitly as
follows. Replace `/path/to/prepared-data` with the prepared data root.

```bash
uv run --frozen --extra inference \
  python -m python_doc_rag chat \
  --data-root /path/to/prepared-data \
  --profile recommended-v1
```

At the checkpoint, `recommended` resolved to Hybrid candidate generation, the
pinned Apache-2.0 mMARCO MiniLM reranker at candidate k 30 / batch 16 / max
length 512, Top-5,
`answer-or-abstain-v1`, and the pinned evaluated Qwen revision in bfloat16.
The historical `profile recommended` command printed the complete profile, and
`check --profile recommended` validated its reused baseline index, metadata,
manifest, and processed JSONL without loading models. Use `recommended-v1` for
those commands on the current checkout.
The reranker, embedding model, index, Qwen model, and tokenizer are each built or
loaded once and reused by `chat`; model-cache/load failures are actionable.

This is an opt-in recommendation, not a production-default change. With no
`--profile`, CLI defaults remain Dense, legacy answer mode, and no reranker;
Hybrid and answer-or-abstain also remain explicit. Full Section Parent is not
connected to `ask` or `chat`. The complete Section implementation and artifacts
are retained for the later general-HTML ingestion work, but that next step should
not assume section parenting won this Python-only tournament.

At the final checkpoint, `uv lock --check`, the 266-test standard and all-extras suites,
both ruff runs, and `git diff --check` passed. The real CLI `check --profile
recommended` validated all 8,677 baseline vectors/metadata records, and a Qwen
chat smoke answered the `re.fullmatch()` question correctly from the official
`re.html#functions` metadata URL before exiting normally. No OpenAI API or OpenAI
Judge call was made anywhere in the sprint.
