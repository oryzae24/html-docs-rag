# 独立holdout Retriever評価

## 固定データセット

- 実行日: 2026-07-29
- 質問数: 18
- SHA-256:
  `f71870a240e7a98ca05716e3e1d8f4d09b2ff856cdd93f608cd6ebeca86ffc36`
- query type: `exact_identifier` 6、`conceptual` 6、`operational` 6
- topic: 17種類。`json`のみ2問で、他16 topicは各1問
- development set: `evaluation/questions.jsonl`の26問

質問はPython 3.13日本語公式文書のprocessed/metadata JSONLを検索し、ページタイトル、
節タイトル、本文、`source_url`を人手確認して作成した。development setと同一ID・
同一質問はなく、単なる語尾変更や直接的な言い換えを避けた。全
`expected_url_keywords`が8,677チャンクの実コーパスに存在することも確認した。

このholdoutはtokenizer、`rrf_k`、`candidate_k`、重み、質問、正解ラベルの調整には
使用しない。結果確認後もこれらを変更していない。

## 実行条件

- Dense: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`と既存FAISS
- Hybrid: Dense＋BM25を同一重みでRRF
- BM25日本語tokenizer: 文字2-gram
- `rrf_k=10`
- `candidate_k=30`
- Top-10
- 詳細JSON（Git管理外）:
  - `<data-root>/evaluation/holdout_dense.json`
  - `<data-root>/evaluation/holdout_hybrid.json`

## 全体結果

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | 平均時間 | 中央時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.5556 | 0.6667 | 0.7778 | 0.8333 | 0.6431 | 0.1967秒 | 0.0088秒 |
| Hybrid | 0.5556 | 0.6667 | 0.7778 | 0.9444 | 0.6492 | 0.1130秒 | 0.0208秒 |

Denseの平均時間は初回queryのwarm-upを含むため、方式間の安定したlatency比較には
中央値または追加warm-up測定が必要である。

## query_type別

| Retriever | query_type | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense | conceptual | 0.5000 | 0.6667 | 0.6667 | 0.8333 | 0.6042 |
| Hybrid | conceptual | 0.5000 | 0.6667 | 0.8333 | 1.0000 | 0.6250 |
| Dense | exact_identifier | 0.8333 | 1.0000 | 1.0000 | 1.0000 | 0.9167 |
| Hybrid | exact_identifier | 0.8333 | 1.0000 | 1.0000 | 1.0000 | 0.9167 |
| Dense | operational | 0.3333 | 0.3333 | 0.6667 | 0.6667 | 0.4083 |
| Hybrid | operational | 0.3333 | 0.3333 | 0.5000 | 0.8333 | 0.4060 |

## topic別Hit@5

| topic | 質問数 | Dense | Hybrid |
| --- | ---: | ---: | ---: |
| argparse | 1 | 0.0000 | 0.0000 |
| asyncio | 1 | 1.0000 | 0.0000 |
| collections | 1 | 1.0000 | 1.0000 |
| contextlib | 1 | 1.0000 | 1.0000 |
| csv | 1 | 1.0000 | 1.0000 |
| dataclasses | 1 | 1.0000 | 1.0000 |
| datetime | 1 | 1.0000 | 1.0000 |
| descriptors | 1 | 0.0000 | 1.0000 |
| functools | 1 | 1.0000 | 1.0000 |
| itertools | 1 | 1.0000 | 1.0000 |
| json | 2 | 1.0000 | 0.5000 |
| logging | 1 | 1.0000 | 1.0000 |
| pathlib | 1 | 1.0000 | 1.0000 |
| regular-expressions | 1 | 0.0000 | 0.0000 |
| tempfile | 1 | 1.0000 | 1.0000 |
| typing | 1 | 0.0000 | 1.0000 |
| unittest | 1 | 1.0000 | 1.0000 |

## 順位変化

- 改善: 5問
- 同順位: 10問
- 悪化: 3問
- Top-5へ入った:
  - `holdout-descriptor-precedence-001`
  - `holdout-typing-protocol-001`
- Top-5から外れた:
  - `holdout-asyncio-taskgroup-001`
  - `holdout-json-ensure-ascii-001`
- Top-10へ入った:
  - `holdout-descriptor-precedence-001`
  - `holdout-re-fullmatch-001`
- Top-10から外れた: なし

DenseのTop-10失敗はdescriptor、argparse、regular expressionの3問だった。Hybridは
descriptorとregular expressionをTop-10へ入れ、argparseの1問だけが失敗した。一方、
HybridはasyncioとJSONの正解をTop-5から押し出した。

## 汎化判断

HybridはholdoutでもHit@10を0.1111、MRR@10を0.0062改善し、Top-10失敗を3問から
1問へ減らした。ただしHit@1、Hit@3、Hit@5はDenseと同値で、operational Hit@5は
0.6667から0.5000へ低下した。改善5問に対し悪化3問もある。

この結果はHybridがTop-10候補集合を広げる可能性を支持するが、既定化を支持するほど
一貫した上位順位改善ではない。Denseを既定のまま維持し、Hybridはopt-inとする。

このholdoutは結果確認済みであり、今後のparameter tuningには使用しない。将来の
release判断には、さらに別の未使用blind setを用意することが望ましい。

## Provenance

- evaluation definition commit:
  `5c2f40dd6fd822f1b65e251441d11469c5e410de`
- source code parent commit:
  `4e608ba6a70413c3db1b027c23ff1e1f2774f4b5`
- holdout questions SHA-256:
  `f71870a240e7a98ca05716e3e1d8f4d09b2ff856cdd93f608cd6ebeca86ffc36`
- Embedding:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Dense: 既存FAISS、Top-10
- Hybrid: Dense＋文字2-gram BM25、`rrf_k=10`、`candidate_k=30`、同一重み
- 実行日: 2026-07-29
