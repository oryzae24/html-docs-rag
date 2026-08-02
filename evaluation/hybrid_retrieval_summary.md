# BM25・Hybrid Retrieval評価

## 実行条件

- 実行日: 2026-07-29
- 評価質問: `evaluation/questions.jsonl`の固定26問
- コーパス: Python 3.13日本語公式ドキュメント、8,677チャンク
- Dense: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`、FAISS
  `IndexFlatIP`、384次元、`cuda:0`
- BM25: プロジェクト内の標準ライブラリ実装、`k1=1.5`、`b=0.75`
- 検索対象: `page_title`、`section_title`、`text`をラベル付き改行で連結
- field別重み付け: なし
- 最終取得数: Top-10
- 詳細JSON（Git管理外）:
  - `<data-root>/evaluation/bm25_baseline.json`
  - `<data-root>/evaluation/hybrid_rrf.json`

Dense index、Embeddingモデル、チャンク設定、評価質問、正解URLは変更していない。
Qwenや回答生成モデルはロードしていない。

## Tokenizer設計

Python識別子は、属性アクセス、アンダースコア、数値添字、空の呼び出し括弧を含む
完全形を保持する。例えば`sys.argv[0]`、`__name__`、`shlex.split()`、
`Path.exists()`を一つのtokenとして残す。属性pathや呼び出しは、表記差を吸収するため
完全形に加えてsuffixと括弧なしの形も追加する。英字の大文字小文字は元表記を保持し、
casefoldしたtokenも追加する。

日本語は空白分割せず、連続する漢字・ひらがな・カタカナから文字n-gramを作る。
コード識別子は文字n-gramへ分解しない。追加の形態素解析依存は使用していない。

## BM25 tokenizer比較

| 日本語tokenizer | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | 平均時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 文字2-gram | 0.1923 | 0.4615 | 0.6923 | 0.6923 | 0.3577 | 0.0086秒 |
| 文字3-gram | 0.0769 | 0.2308 | 0.3846 | 0.5000 | 0.1947 | 0.0051秒 |
| 文字2-gram＋3-gram | 0.1154 | 0.1923 | 0.3846 | 0.5385 | 0.2143 | 0.0098秒 |

2-gramはHit@5、Hit@10、MRR@10のすべてで他候補を上回ったため採用した。
2-gram＋3-gramはtoken数が増えたが、長い一般表現の一致が順位へ強く影響し、改善には
つながらなかった。

## RRF感度

BM25は文字2-gramへ固定し、DenseとBM25を同じ重みで統合した。Dense scoreとBM25
scoreは直接加算していない。

| rrf_k | candidate_k | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 10 | 0.5385 | 0.6154 | 0.6538 | 0.8462 | 0.6036 |
| 60 | 20 | 0.5769 | 0.6538 | 0.6538 | 0.8462 | 0.6342 |
| 60 | 30 | 0.5769 | 0.6538 | 0.6923 | 0.8462 | 0.6353 |
| 30 | 30 | 0.5769 | 0.6538 | 0.6923 | 0.8462 | 0.6353 |
| 10 | 30 | 0.5769 | 0.6538 | 0.6923 | 0.8846 | 0.6389 |

`candidate_k=30`はDenseのHit@5を維持し、`rrf_k=10`はHit@10とMRR@10が最良だった。
このため暫定既定値を`rrf_k=10`、`candidate_k=30`とした。

## 評価セットの位置づけ

この26問はtokenizer方式、`rrf_k`、`candidate_k`の選択にも使用したため、独立した
test setではなくdevelopment benchmarkである。ここでの改善値は方式選択の根拠には
できるが、未知質問への汎化性能を保証しない。

Retrieverの最終構成を確定する前に、この26問と重複せず、パラメータ選択へ使用しない
別のholdout評価セットで再評価する。holdoutの結果が大きく異なる場合は、CLI既定値を
含む採用判断を見直す。

## 全体比較

| 方式 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | 平均時間 | 中央時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.3846 | 0.5385 | 0.6923 | 0.8077 | 0.4972 | 0.0168秒 | 0.0092秒 |
| BM25 | 0.1923 | 0.4615 | 0.6923 | 0.6923 | 0.3577 | 0.0086秒 | 0.0081秒 |
| Hybrid RRF | 0.5769 | 0.6538 | 0.6923 | 0.8846 | 0.6389 | 0.0251秒 | 0.0179秒 |

HybridはDenseのHit@5を維持し、Hit@1を0.1923、Hit@10を0.0769、MRR@10を
0.1417改善した。平均検索時間は約8.3ms増えたが25.1msであり、対話CLIの検索処理と
して許容可能と判断した。BM25単独はexact_identifierを補完する一方、Dense単独の
代替にはならない。

## query_type別比較

| 方式 | query_type | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense | conceptual | 0.2500 | 0.3750 | 0.5000 | 0.7500 | 0.3710 |
| BM25 | conceptual | 0.1250 | 0.2500 | 0.6250 | 0.6250 | 0.2625 |
| Hybrid | conceptual | 0.5000 | 0.5000 | 0.5000 | 0.8750 | 0.5420 |
| Dense | exact_identifier | 0.5000 | 0.7000 | 0.7000 | 0.8000 | 0.5976 |
| BM25 | exact_identifier | 0.2000 | 0.7000 | 0.9000 | 0.9000 | 0.4450 |
| Hybrid | exact_identifier | 0.7000 | 0.8000 | 0.8000 | 0.9000 | 0.7643 |
| Dense | operational | 0.3750 | 0.5000 | 0.8750 | 0.8750 | 0.4979 |
| BM25 | operational | 0.2500 | 0.3750 | 0.5000 | 0.5000 | 0.3438 |
| Hybrid | operational | 0.5000 | 0.6250 | 0.7500 | 0.8750 | 0.5792 |

HybridはconceptualのHit@5を維持しつつMRRを0.3710から0.5420へ改善した。
exact_identifierはすべての主要指標がDense以上だった。operationalはMRRとHit@1/3を
改善したが、Hit@5は0.8750から0.7500へ低下したため、質問別退行を継続監視する。

## topic別Hit@5

| topic | 質問数 | Dense | BM25 | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| entrypoint | 2 | 0.0000 | 0.5000 | 0.0000 |
| exceptions | 2 | 0.5000 | 1.0000 | 0.5000 |
| io | 4 | 0.5000 | 0.5000 | 0.5000 |
| mappings | 1 | 1.0000 | 1.0000 | 1.0000 |
| os | 2 | 0.5000 | 0.5000 | 0.5000 |
| pathlib | 2 | 1.0000 | 1.0000 | 1.0000 |
| readline | 1 | 1.0000 | 0.0000 | 1.0000 |
| sequences | 1 | 1.0000 | 1.0000 | 1.0000 |
| shlex | 2 | 1.0000 | 1.0000 | 1.0000 |
| subprocess | 3 | 1.0000 | 0.6667 | 0.6667 |
| sys | 2 | 0.5000 | 0.5000 | 0.5000 |
| threading | 1 | 1.0000 | 1.0000 | 1.0000 |
| unittest | 3 | 0.6667 | 0.6667 | 1.0000 |

## 注目質問

| 質問 | Dense順位 | BM25順位 | Hybrid順位 | 結果 |
| --- | ---: | ---: | ---: | --- |
| main関数はどこに配置するべきですか？ | 圏外 | 5 | 9 | Top-10へ入ったがTop-5外 |
| isatty()とは何ですか？ | 圏外 | 圏外 | 圏外 | 未改善 |
| EOFErrorはどのような場合に発生しますか？ | 圏外 | 3 | 7 | Top-10へ改善 |
| 標準出力はいつ行／ブロックバッファリングされますか？ | 圏外 | 圏外 | 圏外 | 未改善 |
| Pythonでパイプ用ファイルディスクリプタを作るには？ | 圏外 | 圏外 | 圏外 | 未改善 |

`isatty`の完全token自体は保持されているが、「とは何ですか」の一般的な日本語
2-gramと文書長の影響が強く、正解の`io`・`os`節はTop-10へ入らなかった。後続の
field別重み付けやRerank候補として残し、この評価で恣意的な重みは追加しない。

## 主な改善と悪化

Hybridで改善した例:

- `shlex.split()` 2位→1位
- `setUp()`・`tearDown()` 7位→1位
- `KeyboardInterrupt` 3位→1位
- `EOFError` 圏外→7位
- main関数の配置 圏外→9位
- daemon thread 2位→1位
- `read()`と`readline()` 7位→1位
- subprocess標準出力取得 3位→1位
- shlexによるコマンド分割 4位→1位

Hybridで悪化した例:

- `os.pipe()` 1位→2位
- `shell=True`の安全性 5位→10位
- ファイルの1行ずつ読み込み 5位→10位
- readline履歴保存 1位→3位

Top-10失敗はDenseの5問からHybridの3問へ減った。Hybridの失敗は`isatty`、標準出力
バッファリング、自然文のpipe作成である。

## 暫定採用判断

Hybrid RRFはCLIの既定Retrieverへはまだ採用せず、明示的なopt-inとして残す。

検索単体ではHit@5を維持し、MRR@10とHit@1/10が改善した。一方、この26問は設定選択
にも使用したdevelopment benchmarkであり、独立した検証ではない。さらに10問の実
Qwen比較では正解URLのTop-5 HitがDense、Hybridとも5/10で、Hybrid側に検索根拠と
回答内容が一致しない例があった。引用形式の検証を通過したことだけでは、回答の意味的
な根拠整合性を保証できない。

CLIは`--retriever dense`を既定とし、`--retriever hybrid`で比較できるようにする。
別のholdout質問セットと回答の意味的grounding評価で改善を確認してから既定化を
再検討する。Parent Document、Rerank、MMR、Query Rewrite、回答不能判定は実装して
いない。

## Provenance

- Source code parent commit: `5ad4e002da6722fb11a8b4846816074a795778ad`
- questions SHA-256:
  `363ed4d55564ba3b925a90100bc002dcf9ffe462a44ec899184323fd5adc2f40`
- Dense JSON SHA-256:
  `889585cab61b8db46b89321164f13509f9196af3152228b18a8c2c4a25e89d9c`
- BM25 JSON SHA-256:
  `dd8b064bcb483b6bf9f5ce0a631396a1002b4e0d5b645b3422b365a5be7b1d9f`
- Hybrid JSON SHA-256:
  `da8db8b74413e70c52e3f34a2693a0e473402eea5d6fdad27d13225b02f1934b`
- Python: 3.12.3
- Sentence Transformers: 5.6.1
- FAISS: 1.14.3
- 実行device: `cuda:0`

評価実装は未コミット状態で測定したため、確定commitは後続のコミット時に追加する。
入力と保存結果の同一性は上記SHA-256でも確認できる。
