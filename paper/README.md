# UBE paper analyses

Analysis code for [**Universal Byte-Level Encoding: UTF-8/UTF-16 Routing to
Reduce Cross-Script Token-Budget Disparities**](https://arxiv.org/abs/2610.01984)
(NeurIPS 2026), by Hyunsik Kim and Youngmoon Jung, Samsung Research.

The tokenizer implementation and usage examples are in the
[repository README](../README.md). This directory contains the following
analyses and their results.

## Contents

| Component | Location |
|---|---|
| FLORES-200 devtest download and check | `scripts/prepare_flores.py` |
| Token counts, token premiums and Gini on FLORES-200 | `scripts/measure_flores.py` |
| Unicode 17 round-trip audit of UBE, SCRIPT-BPE and MYTE | `scripts/unicode_audit.py` |
| MYTE-UBE transport, its parity verification and base-symbol counts | `scripts/myte_ube.py`, `scripts/myte_codec.py` |
| Decoding of token sequences from the middle of a sentence: streams, detokenizer windows and slices | `scripts/measure_decoding.py` |
| Outputs of the commands below | `results/` |

## Installation

Python 3.10 or newer and a [Rust toolchain](https://rustup.rs) are needed to
build the tokenizer from source. Use a fresh virtual environment, since the
build installs as the `tokenizers` package (version 0.23.3). The Unicode audit
also needs [uv](https://docs.astral.sh/uv/).

From the root of a clone of [Samsung/UBE](https://github.com/Samsung/UBE):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ./bindings/python -r paper/requirements.txt
cd paper
```

Run all commands below from `paper/`. Data, external checkouts and outputs
are relative to this directory.

## Reproducing the results

### FLORES-200 data

```bash
python scripts/prepare_flores.py
```

This downloads the official FLORES-200 archive, checks its SHA-256 digest, and
writes the devtest text of the 204 languages (1,012 sentences each) to
`data/flores200_devtest.json`. The text is checked against the digest of the
copy used in the paper. FLORES-200 is licensed under CC BY-SA 4.0 and is not
redistributed here.

### Token counts and disparity

```bash
python scripts/measure_flores.py path/to/tokenizer.json [more tokenizer.json files] --output-dir results/flores200
```

For each tokenizer, the script records the total and mean token count of every
language. A language's token premium is its mean tokens per sentence
divided by that of English (`eng_Latn`); the summary reports the mean, median
and maximum premium and the Gini coefficient of the 203 non-English premiums.
Padding and truncation are disabled during measurement; special tokens added
by the tokenizer's post-processor are included.

### Unicode 17 round-trip audit

The audit runs the official SCRIPT-BPE and MYTE implementations at pinned
commits. SCRIPT-BPE runs in its own environment through
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tomlimi/MYTE
git -C MYTE checkout 177299ca7278a3618b8c77588e6dcd3912bad428
git clone https://github.com/sanderland/script_tok
git -C script_tok checkout 54a105829f302fe3a3fc5e963f150ba7d839fb53

python scripts/unicode_audit.py --myte-dir MYTE --script-tok-dir script_tok
```

The audit downloads the Unicode 17 data files and checks their digests. It then
round-trips every Unicode scalar value and every unique sequence in
`NormalizationTest.txt`, `GraphemeBreakTest.txt` and `emoji-test.txt`. Raw
identity and identity after each method's configured normalization (NFC for
SCRIPT-BPE) are reported separately. Round-trip exactness depends on the base
representation and normalization rather than on learned merges. By default the
audit therefore trains small UBE and SCRIPT-BPE tokenizers (4,096 merges each)
on `data/flores200_devtest.json`; `--ube-tokenizer` and `--script-tokenizer`
audit given tokenizer files instead, and auditing the 32K tokenizers used in
the paper gives identical counts.

| Unicode 17 input set | Inputs | UBE raw exact | SCRIPT-BPE after NFC | MYTE raw exact |
|---|---:|---:|---:|---:|
| Assigned through Unicode 16 (excluding Private-Use) | 155,063 | **All recovered** | **All recovered** | 154,023/155,063 |
| Assigned in Unicode 17 | 4,803 | **All recovered** | Outside released coverage | **All recovered** |
| Private-Use (Co) | 137,468 | **All recovered** | Filtered by design | **All recovered** |
| Unassigned/reserved (Cn) | 814,730 | **All recovered** | Filtered by design | **All recovered** |
| `NormalizationTest.txt` unique inputs | 37,671 | **All recovered** | 166 outside coverage | 21,373/37,671 |
| `GraphemeBreakTest.txt` unique inputs | 764 | **All recovered** | 74 outside coverage | 684/764 |
| `emoji-test.txt` unique inputs | 5,225 | **All recovered** | 67 outside coverage | **All recovered** |
| All suite-level cases | 1,155,724 | **All recovered** | No aggregate across excluded inputs | 1,138,306/1,155,724 |

SCRIPT-BPE's character coverage comes from the Unicode 16 script table bundled
with its repository, and its NFC normalization from the Unicode database of the
Python interpreter in its environment (Python 3.13, pinned by its repository,
with Unicode 15.1). The full report is in `results/unicode_audit.md`.

### MYTE-UBE

```bash
python scripts/myte_ube.py --myte-dir MYTE
```

MYTE-UBE applies MYTE's released decomposition and morphology-merge maps
unchanged and routes only the characters that no merge rule rewrote through
UBE. Its 512 symbols form two regions: bytes that are copied verbatim into the
post-MYTE byte stream (UTF-8 bytes of residual characters and MYTE code bytes),
and routed UTF-16LE bytes. For every FLORES-200 devtest sentence, the script
checks that the symbols rebuild the post-MYTE byte stream exactly and that the
decoded text equals the output of MYTE's released implementation
(`src/rewrite_bytes.py::ByteRewriter`). It then reports base symbols per
sentence. No BPE is applied, and one symbol is one model-input position:

| Representation | Alphabet | English | Korean, Chinese, Japanese | Other BMP (7) | Overall | Total positions | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|
| UBE | 512 | 130.481 | **91.252** | 277.056 | 176.088 | 36,353,040 | 0.1500 |
| UTF-8 bytes | 256 | 130.530 | 133.665 | 406.632 | 196.106 | 40,485,643 | 0.2077 |
| MYTE-UBE | 512 | **109.094** | 94.610 | **247.622** | **141.354** | **29,182,333** | **0.0925** |
| MYTE | 256 | 109.140 | 124.612 | 349.398 | 148.142 | 30,583,555 | 0.1177 |

Relative to MYTE, MYTE-UBE reduces the Korean, Chinese and Japanese mean by
24.08%, the Other BMP mean by 29.13% and total positions by 4.58%. It is lower
in 201 of the 204 languages and tied in the remaining 3. Korean, Chinese and
Japanese cover the languages tagged `Hang`, `Hans`, `Hant` and `Jpan` (Chinese
in both scripts and Cantonese); Other BMP (7) covers `Tibt`, `Olck`, `Tfng` and
`Mymr`. The base-symbol counts and parity checks are in
`results/myte_ube_flores.json`.

Against UTF-8 bytes, UBE alone shortens the Korean, Chinese and Japanese mean by
31.73%, the Other BMP (7) mean by 31.87% and total positions by 10.21%. It is
shorter in 202 of the 204 languages, ties in Catalan and Central Kurdish, and is
longer in none.

The Gini column reports the population Gini coefficient of the 203 non-English
base-symbol premiums. For each representation, a language's premium is its
mean symbols per sentence divided by that of `eng_Latn`. Each language has
1,012 sentences, so the ratios of the per-language totals in
`results/myte_ube_flores.json` equal the ratios of the means. These totals give
0.1500 for UBE and 0.2077 for UTF-8 bytes, and 0.0925 for MYTE-UBE and 0.1177 for
MYTE.

### Partial decoding

```bash
python scripts/measure_decoding.py path/to/tokenizer.json [more tokenizer.json files] --output-dir results/decoding
```

For each tokenizer, every devtest sentence is encoded without special tokens
and split at the character boundary between tokens nearest the token-sequence
midpoint, choosing the earlier boundary on ties. The first part serves as a
prompt, and the rest is decoded incrementally by `decoders.DecodeStream`
started with the prompt ids, and by a detokenizer that
decodes a window of tokens and emits the text added by the newest ones. Its
first window starts five tokens before the end of the prompt, or at
`UniversalByteLevelDecoding.aligned_offset` of that position; a single decode
of the whole continuation starts one token before the end of the prompt, or at
the aligned offset of that position. The token ids from every interior token
position to the end of the sentence are also decoded, with `Tokenizer.decode`
and with `UniversalByteLevelDecoding.decode_slice`; such a slice counts as exact
when its text is the following characters, preceded by one or more U+FFFD for a
character that the cut divides. The script records the number of exact decodes
of each kind per language. Languages run in separate workers; each worker uses
sequential tokenization. Window-width summaries are null when there are no
eligible streams.

### Tests

```bash
pip install pytest
MYTE_DIR=MYTE pytest tests
```

## Citation

See the [repository citation](../README.md#citation).

## License

Copyright 2026 Samsung Electronics Co., Ltd.

The code in this directory is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0); see
[LICENSE-CC-BY-NC-SA-4.0](../LICENSE-CC-BY-NC-SA-4.0). FLORES-200, the Unicode data files, the MYTE maps and SCRIPT-BPE are
obtained from their original sources at run time and are not redistributed
here.
