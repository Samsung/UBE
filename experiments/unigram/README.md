<!--
Copyright 2026 Samsung Electronics Co., Ltd.
SPDX-License-Identifier: CC-BY-NC-SA-4.0
-->

# UBE with Unigram tokenizers

This directory reproduces the [Unigram tokenizers](../../README.md#unigram-tokenizers)
results of the repository README. Unigram and BPE tokenizers are trained with
UBE and with UTF-8 bytes (the 256 symbols of `ByteLevel`) on the same samples
of the 101-language mC4 training mixture of the paper, and measured on
FLORES-200 devtest.

Each training setting, a sample size with a vocabulary size, has four
tokenizers trained on the same text: UBE and UTF-8, each with the Unigram and
the BPE model. The UBE and UTF-8 tokenizers of a model share the
trainer settings and the training text, and differ only in the base alphabet,
with its pre-tokenizer, post-processor and decoder. The README table reports
UBE's change relative to UTF-8, `100 × (UBE / UTF-8 − 1)`, in three
quantities:

- the mean tokens per sentence of the 39 languages written mostly in 3-byte
  characters: the FLORES-200 languages in which at least half of the
  characters (Unicode scalar values, spaces included) take 3 or more UTF-8
  bytes, averaged with equal weight per language;
- the mean tokens per English sentence;
- the Gini coefficient of the 203 non-English token premiums, a premium being
  a language's mean tokens per sentence divided by that of English.

## Contents

| Component | Location |
|---|---|
| Byte shares of the paper's 101-language training mixture | `mixture_shares.json` |
| Training-text sampling with those shares | `scripts/sample_corpus.py` |
| Tokenizer training: UBE or UTF-8, with Unigram or BPE | `scripts/train_tokenizer.py` |
| FLORES-200 devtest download and token counts | `../../paper/scripts/prepare_flores.py`, `../../paper/scripts/measure_flores.py` |
| UBE's change from UTF-8 and the README table | `scripts/compare.py` |
| Tests on small synthetic data | `tests/` |

## Installation

Python 3.10 or newer and a [Rust toolchain](https://rustup.rs) are needed to
build the tokenizer from source. Use a fresh virtual environment, since the
build installs as the `tokenizers` package (version 0.23.3).

From the root of a clone of [Samsung/UBE](https://github.com/Samsung/UBE):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ./bindings/python -r paper/requirements.txt
cd experiments/unigram
```

Run all commands below from `experiments/unigram/`. Samples, tokenizers and
results are written relative to this directory.

## Reproducing the results

### Training text

Each language needs its mC4 training documents as JSON Lines files, one
document per line in a `text` field, as in the mC4 distribution. Files may be
gzip-compressed (`.gz`). `--input` is a glob pattern in which `{lang}` stands
for the mC4 language code of `mixture_shares.json`. The files that match a
language are read in sorted path order, so name them in the order of the
training shards. The commands below read each language's files from
`mc4/<lang>/`.

```bash
python scripts/sample_corpus.py --input 'mc4/{lang}/*.json.gz' --total-bytes 1073741824 --output-dir corpus/1gib
python scripts/sample_corpus.py --input 'mc4/{lang}/*.json.gz' --total-bytes 4294967296 --output-dir corpus/4gib
```

Each language receives the first `round(T × share)` UTF-8 bytes of its
documents, where `T` is `--total-bytes` and the share is the language's value
in `mixture_shares.json` divided by `total_bytes`: its share of the paper's
mC4 tokenizer-training data, measured in stored Arrow data. Empty documents are
skipped, documents are taken whole, and the document that would cross the
quota is cut at the last character boundary at or below it. The sampling
involves no randomness, so the 1 GiB sample is a per-language prefix of the
4 GiB sample of the same input. The script writes `<lang>.jsonl` for each
language and `manifest.json` with each language's quota, text bytes,
documents, input files and SHA-256.

### Tokenizers

The README table uses vocabulary sizes of 16,000, 32,000 and 64,000 with the
1 GiB sample and 32,000 with the 4 GiB sample, four tokenizers each:

```bash
export RAYON_NUM_THREADS=14 TOKENIZERS_PARALLELISM=true
for setting in 1gib:16000 1gib:32000 1gib:64000 4gib:32000; do
  sample=${setting%:*} vocab=${setting#*:}
  for kind in ube_unigram bytelevel_unigram ube_bpe bytelevel_bpe; do
    python scripts/train_tokenizer.py --corpus corpus/$sample --kind $kind --vocab-size $vocab \
        --output tokenizers/${sample}_$vocab/$kind/tokenizer.json
  done
done
```

The documents are passed to `Tokenizer.train_from_iterator` one sequence per
document, with the languages in code order and each language's documents in
sample order. Each `tokenizer.json` is saved with a `tokenizer.meta.json` that
records its trainer arguments, thread settings, the digest of the sample
manifest and the library version. With 14 threads, the 16 tokenizers took
about six hours to train. On the 4 GiB sample, a Unigram tokenizer took up to
1.5 hours and a BPE tokenizer up to about 60 GiB of memory.

### FLORES-200 token counts

The FLORES-200 data and measurement come from [`paper/`](../../paper/README.md):

```bash
python ../../paper/scripts/prepare_flores.py
for run in 1gib_16000 1gib_32000 1gib_64000 4gib_32000; do
  python ../../paper/scripts/measure_flores.py tokenizers/$run/*/tokenizer.json --output-dir results/$run
done
```

`prepare_flores.py` downloads the official FLORES-200 archive, checks its
digest and writes the devtest text of the 204 languages (1,012 sentences each)
to `../../paper/data/flores200_devtest.json`, the default input of
`measure_flores.py`. For each tokenizer, `measure_flores.py` writes each
language's total and mean token counts, along with summary statistics for the
non-English premiums, including their Gini coefficient. The output file takes
the tokenizer directory's name, for example `ube_unigram.json`.

### UBE's change from UTF-8

```bash
python scripts/compare.py \
    --setting "1 GiB text, 16K" results/1gib_16000 \
    --setting "1 GiB text, 32K" results/1gib_32000 \
    --setting "1 GiB text, 64K" results/1gib_64000 \
    --setting "4 GiB text, 32K" results/4gib_32000 \
    --output results/comparison.json
```

Each `--setting` gives a row label and the measurement directory of its four
tokenizers. The script derives the 39-language group from the FLORES-200 text
and prints the README table:

| UBE vs UTF-8 (FLORES-200 devtest) | Unigram: 39 languages | BPE: 39 languages | Unigram: English | Unigram: Gini |
| :-------------------------------- | --------------------: | ----------------: | ---------------: | ------------: |
| 1 GiB text, 16K                   |                −15.1% |             −7.5% |            −7.3% |        −15.5% |
| 1 GiB text, 32K                   |                −16.4% |             −8.2% |            −8.8% |        −15.8% |
| 1 GiB text, 64K                   |                −17.8% |             −5.3% |            −9.6% |        −15.7% |
| 4 GiB text, 32K                   |                −16.1% |             −6.7% |           −10.3% |        −14.9% |

`--output` writes the 39-language mean, the English mean, the Gini coefficient
and the total tokens over the 204 languages for both models and both base
alphabets, each with UBE's change.

## Settings of the README table

| Training text | 1 GiB sample | 4 GiB sample |
|---|---:|---:|
| `--total-bytes` | 1,073,741,824 | 4,294,967,296 |
| Text bytes | 1,073,741,791 | 4,294,967,269 |
| Documents | 260,149 | 1,034,666 |
| Languages | 101 | 101 |
| English share of the text bytes | 29.24% | 29.24% |
| Vocabulary sizes | 16,000, 32,000, 64,000 | 32,000 |

Both samples were drawn from the paper's mC4 training data, reading each
language's training shards in order. Vocabulary sizes include the special
tokens `<s>`, `<pad>`, `</s>`, `<unk>` and `<mask>`, which take ids 0 to 4.

| Tokenizer component | UBE | UTF-8 (`ByteLevel`) |
|---|---|---|
| Normalizer | none | none |
| Pre-tokenizer | `UniversalByteLevel(add_prefix_space=False, use_regex=True, trim_offsets=False)` | `ByteLevel(add_prefix_space=False)` |
| Post-processor | `UniversalByteLevel(trim_offsets=False)` | `ByteLevel(trim_offsets=False)` |
| Decoder | `UniversalByteLevel()` | `ByteLevel()` |
| Initial alphabet | `UniversalByteLevel.alphabet()`, 512 symbols | `ByteLevel.alphabet()`, 256 symbols |

Both pre-tokenizers split text with the GPT-2 pattern.

| Trainer | Arguments |
|---|---|
| Unigram | `UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens, initial_alphabet=initial_alphabet, unk_token="<unk>", max_piece_length=16, show_progress=False)` |
| BPE | `BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens, initial_alphabet=initial_alphabet, show_progress=False)` |

Every other trainer argument keeps its library default; for Unigram these are
`shrinking_factor=0.75`, `n_sub_iterations=2` and `seed_size=1000000`. The
maximum piece length of 16 counts base symbols.

`prepare_flores.py` writes the devtest text used for the table: 204 languages
with 1,012 sentences each. Token counts include any special tokens added by
the post-processor; the UBE and `ByteLevel` post-processors add none.

## Tests

```bash
pip install pytest
pytest tests
```

The tests sample, train and measure small synthetic corpora and download
nothing.

## License

Copyright 2026 Samsung Electronics Co., Ltd.

The code in this directory is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0); see
[LICENSE-CC-BY-NC-SA-4.0](../../LICENSE-CC-BY-NC-SA-4.0). mC4 and FLORES-200 are
obtained from their original sources and are not redistributed here.
