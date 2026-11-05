# Tokenizers 1.0 with Universal Byte-Level Encoding (UBE)

This folder is Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers)
v1.0.0-rc.2, the 1.0 release candidate, with the Universal Byte-Level Encoding
(UBE) from
[**Universal Byte-Level Encoding: UTF-8/UTF-16 Routing to Reduce Cross-Script
Token-Budget Disparities**](https://arxiv.org/abs/2610.01984) (NeurIPS 2026).
The repository root holds UBE on `tokenizers` v0.23.3; see the
[repository README](../README.md).

UBE is a byte representation for byte-level BPE that keeps BPE training and
merges standard. A character of one or two UTF-8 bytes keeps its UTF-8 bytes; a
character of three or four UTF-8 bytes is written with the bytes of its UTF-16LE
code units, at U+E100 + byte, for a 512-symbol base alphabet.

## Installation

Use Python 3.10 or newer and a [Rust toolchain](https://rustup.rs) to build
from source, in a fresh virtual environment: the package keeps the name and
version of its upstream base (`tokenizers` 1.0.0rc2).

```bash
python -m venv .venv
source .venv/bin/activate
pip install "git+https://github.com/Samsung/UBE@v0.23.3-ube.2#subdirectory=v1/bindings/python"
```

## Usage

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
encoding = tokenizer.encode("Hi 빙고 🩷💚", add_special_tokens=False)
assert tokenizer.decode(encoding.ids) == "Hi 빙고 🩷💚"
```

A UBE `tokenizer.json` loads like any other: one trained with the example
below, or with `UniversalByteLevelBPETokenizer` and its default options in the
repository root.

## Training

The `train_universal_byte_level` example of `tk-train` trains a UBE tokenizer
and writes it as a `tokenizer.json`:

```bash
cd v1/tokenizers/tk-train
cargo run --release --example train_universal_byte_level -- \
    --vocab-size 32000 \
    --special-token "<s>" --special-token "<pad>" --special-token "</s>" \
    --special-token "<unk>" --special-token "<mask>" \
    --output tokenizer.json corpus.txt
```

Each line of the input files is one training sequence, including its line
ending, and `--min-frequency` defaults to 2. The example starts from the same
512-symbol alphabet, with the same default training parameters, as
`UniversalByteLevelBPETokenizer.train` in the repository root.

## What this folder adds

- A `universal_byte_level` flag on the BPE model (`tk-encode`), written as
  `"universal_byte_level": true` in `tokenizer.json` (`tk-serialize`). The
  vocabulary is spelled in UBE's 512-symbol alphabet: the 256 byte symbols of
  byte-level BPE for UTF-8 bytes and U+E100 to U+E1FF for UTF-16LE bytes.
- `tk_encode::utils::universal_byte_level`, with the alphabet (`alphabet`) and
  the spelling of text in it (`spell`).
- Conversion of a `tokenizer.json` with the `UniversalByteLevel` pre-tokenizer
  of the repository root to that flag (`tk-convert`).
- UBE training with `tk-train`: the `train_universal_byte_level` example.

For the library's full API, see the upstream
[README](https://github.com/huggingface/tokenizers/blob/v1.0.0-rc.2/README.md).

## License

This folder is a modified version of Hugging Face `tokenizers` v1.0.0-rc.2. The
Hugging Face code keeps its Apache License 2.0; see `LICENSE`. The UBE
modifications and additions are Copyright 2026 Samsung Electronics Co., Ltd. and
are licensed under CC BY-NC-SA 4.0; see
[`LICENSE-CC-BY-NC-SA-4.0`](../LICENSE-CC-BY-NC-SA-4.0) and [`NOTICE`](../NOTICE). To
cite UBE, see the [repository README](../README.md#citation).
