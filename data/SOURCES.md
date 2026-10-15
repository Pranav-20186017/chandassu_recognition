# Corpus and metre-rule sources

The active eight-class research dataset is [v3](v3/README.md); its complete retained
work inventory and source links are in [v3/WORKS.md](v3/WORKS.md), with per-class
counts and partition roles in [v3/works.json](v3/works.json). Historical four-class
catalogues remain [v1/WORKS.md](v1/WORKS.md) and [v2/WORKS.md](v2/WORKS.md).
Source URLs establish attribution, not blanket text redistribution permission.
See [LICENSE.md](LICENSE.md).

## Rules checked on 15 October 2026

| Metre | Structure relevant to this experiment | Reference |
|---|---|---|
| ఆటవెలది | Four padas. Padas 1/3: three Surya then two Indra ganas; 2/4: five Surya ganas. Yati at first syllable of gana 4; prasa-yati allowed; no mandatory prasa. | [Telugu Wikipedia](https://te.wikipedia.org/wiki/ఆటవెలది) |
| తేటగీతి | Four padas, each Surya + Indra + Indra + Surya + Surya. Yati at gana 4; prasa-yati allowed; no mandatory prasa. | [Telugu Wikipedia](https://te.wikipedia.org/wiki/తేటగీతి) |
| కందము | Four padas, with three four-matra ganas in 1/3 and five in 2/4. Allowed ganas are గగ, భ, జ, స, నల, with positional restrictions on జ. The third gana of 2/4 is జ or నల; even padas end in guru. Opening syllable weight agrees across padas. Yati in 2/4 at gana 4; prasa required, prasa-yati does not replace it. | [Telugu Wikipedia](https://te.wikipedia.org/wiki/కందం) |
| సీసము | Four long body padas, each six Indra + two Surya ganas. Often displayed as eight alternating half-padas (four Indra; two Indra + two Surya). Yati between ganas 1/3 and 5/7; prasa-yati allowed; no mandatory prasa. A following Ataveladi/Tetageeti closes the composition. Body-only classification is the chosen experiment. | [Telugu Wikipedia](https://te.wikipedia.org/wiki/సీసము_(పద్యం)), [ఆంధ్రనాటక పద్యపఠనం, primary text on Wikisource](https://te.wikisource.org/wiki/ఆంధ్రనాటక_పద్యపఠనం/పద్యరూపం) |

Wikipedia is orientation for representation, not a validated machine verifier or
annotation authority for individual poems. Ataveladi/Kandam pages have editorial
quality cautions. The CNN receives raw poem text and source labels; these rule
summaries do not become features or intermediate supervision.

## Acquisition evidence

V3 uses only existing local acquisitions; no new live scrape is necessary:

- Frozen `data/v2/{train,validation,test}.jsonl` plus `splits/train_expansion.json`.
- `data/v1/other.jsonl`: accepted independent poems with explicit non-target markers
  preserved in `label_notes`; labels are mapped only through unambiguous markers.
- `archive/data/andhrabharati/*/report.json` and their saved `raw/*.html`: labelled
  non-vrutta verse cells previously skipped by the four-class scraper. Page source
  URLs, raw HTML hashes and existing source/author attribution are retained.
- `archive/data/expansion_20261011/extraction.jsonl`: accepted source annotations.
- `archive/data/{independent_poetry/v1,independent_poetry/v2,expansion_20261011}/review.jsonl`:
  explicit 8+4 Seesamu compound records, now admitted for body-only representation.

Raw/review pools contain duplicates and unsupported layouts. Their counts are not
training counts. The authoritative unique retained totals and exclusions are in
[v3/manifest.json](v3/manifest.json) and [v3/audit.json](v3/audit.json). Ambiguous
`గీ.` remains unassigned; no rule-engine inference is used to inflate Tetageeti.
Every poem stays whole and every work/edition/exact-line group stays in one role.
