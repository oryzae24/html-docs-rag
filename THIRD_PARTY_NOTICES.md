# Third-party notices

## Python Documentation

The project includes an unchanged-content, deterministically packaged source
snapshot of the Japanese Python 3.13 documentation under
`resources/source_snapshots/python-3.13-ja-2026-07-20.zip`.

- Upstream source: `https://docs.python.org/ja/3.13/archives/python-3.13-docs-html.zip`
- Acquisition timestamp recorded with the retained expanded tree:
  `2026-07-20T20:34:21.446060+00:00`
- Original archive SHA-256 recorded at acquisition:
  `f8ddb3454726cbe34580b4c21723128a1b33b50f1155e9b9184cb790db66d9cb`
- Deterministic project snapshot SHA-256:
  `1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777`
- Provenance:
  `resources/source_snapshots/python-3.13-ja-2026-07-20.provenance.json`

The original archive bytes were no longer available when this repository
snapshot was made. The project snapshot was produced from the retained expanded
HTML tree after that tree reproduced the protected chunk corpus byte-for-byte.
It is therefore not represented as having the original recorded archive SHA.
The source files' contents were not modified when they were packaged.

The upstream license and copyright documents remain inside the snapshot at:

- `python-3.13-docs-html/license.html`
- `python-3.13-docs-html/copyright.html`
- `python-3.13-docs-html/_sources/license.rst.txt`
- `python-3.13-docs-html/_sources/copyright.rst.txt`

Those files are the authoritative license and copyright notices for the
included documentation content.

## Hugging Face models

The repository configuration refers to Hugging Face models for embedding,
reranking, and local answer generation. Model weights are not included in this
repository. Users who download a configured model must comply with that model's
license and any associated usage terms published by its provider.

## Relationship to the project license

The MIT License in `LICENSE` applies to original project code. It does not
replace, override, or relicense the Python documentation snapshot, configured
Hugging Face models, or any other third-party material. The notices and license
files supplied by each upstream project remain controlling for that material.
