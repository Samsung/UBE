# Unicode 17 round-trip audit

A raw-exact case satisfies `decode(encode(x)) == x` at the Unicode-scalar-sequence level. Exactness after each method's configured normalizer is reported separately: identity for UBE and MYTE, and NFC for SCRIPT-BPE. Unicode 17 NFC-exactness is also reported where the standard supplies an expected output. The audit is deterministic and contains no sampled accuracy estimate.

## Exhaustive scalar coverage

| Unicode 17 stratum | # scalars | UBE raw exact | MYTE raw exact | SCRIPT-BPE raw exact | SCRIPT-BPE exact after configured NFC |
|---|---:|---:|---:|---:|---:|
| Assigned through Unicode 16 (excluding Private-Use) | 155,063 | 155,063/155,063 | 154,023/155,063 | 153,943/155,063 | 155,063/155,063 |
| Assigned in Unicode 17 | 4,803 | 4,803/4,803 | 4,803/4,803 | 0/4,803 | 0/4,803 |
| Private-Use (Co) | 137,468 | 137,468/137,468 | 137,468/137,468 | 0/137,468 | 0/137,468 |
| Unassigned/reserved (Cn) | 814,730 | 814,730/814,730 | 814,730/814,730 | 0/814,730 | 0/814,730 |
| All Unicode scalar values | 1,112,064 | 1,112,064/1,112,064 | 1,111,024/1,112,064 | 153,943/1,112,064 | 155,063/1,112,064 |

## Official Unicode sequence suites

| Suite | # unique inputs | UBE raw exact | MYTE raw exact | SCRIPT-BPE raw exact | SCRIPT-BPE exact after configured NFC | SCRIPT-BPE exact against Unicode 17 NFC |
|---|---:|---:|---:|---:|---:|---:|
| NormalizationTest.txt | 37,671 | 37,671/37,671 | 21,373/37,671 | 20,922/37,671 | 37,505/37,671 | 37,385/37,671 |
| GraphemeBreakTest.txt | 764 | 764/764 | 684/764 | 667/764 | 690/764 | n/a |
| emoji-test.txt | 5,225 | 5,225/5,225 | 5,225/5,225 | 5,158/5,225 | 5,158/5,225 | n/a |

## Methodology

- Lossless-tokenization criterion: raw identity is reported separately from equality after each method's configured normalization.
- Scalar universe: every value from U+0000 through U+10FFFF except the surrogate range U+D800-U+DFFF.
- Sequence suites: every unique sequence in the Unicode 17 normalization, grapheme-break, and emoji test files.
- NormalizationTest.txt supplies normative NFC targets. The grapheme-break and emoji files are used only as official multi-scalar stress inputs, not as tokenizer conformance oracles.
- Inputs and tokenizer artifacts are fixed by SHA-256 in the machine-readable JSON.
- Tokenizers: UBE was trained with 4,096 merges on `data/flores200_devtest.json`; SCRIPT-BPE was trained with 4,096 merges on `data/flores200_devtest.json`. Round-trip exactness depends on the base representation and normalization, not on learned merges.
- MYTE directly uses the official `src/rewrite_bytes.py::ByteRewriter` for both rewrite passes and directions at the pinned repository commit; the implementation and both map files are fixed by SHA-256 in the JSON.

## References used to define the audit

1. [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/). EMNLP System Demonstrations 2018. Separates reversible tokenization after configured normalization from recovery of the original raw string.
2. [Neural Machine Translation with Byte-Level Subwords](https://doi.org/10.1609/aaai.v34i05.6451). AAAI 2020. Motivates exhaustive coverage because byte-level vocabularies are intended to avoid out-of-vocabulary inputs.
3. [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://aclanthology.org/2022.tacl-1.17/). TACL 2022. Motivates testing raw UTF-8 coverage without language-specific preprocessing.
4. [Unicode Standard Annex #15: Unicode Normalization Forms](https://www.unicode.org/reports/tr15/). Unicode 17.0.0. Supplies the normative normalization conformance cases and expected NFC outputs.
5. [Unicode Standard Annex #29: Unicode Text Segmentation](https://www.unicode.org/reports/tr29/). Unicode 17.0.0. Supplies official multi-scalar grapheme-break test sequences.
6. [Unicode Standard Annex #44: Unicode Character Database](https://www.unicode.org/reports/tr44/). Unicode 17.0.0. Defines the category and age property files used to construct the exhaustive scalar strata.
7. [Unicode Technical Standard #51: Unicode Emoji](https://www.unicode.org/reports/tr51/). Unicode Emoji 17.0. Supplies the versioned emoji sequence test set.
