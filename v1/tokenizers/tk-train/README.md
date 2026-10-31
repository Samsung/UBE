<!--
Modified by Samsung Electronics Co., Ltd. in 2026 for Universal Byte-Level Encoding (UBE).
Samsung's modifications are licensed under CC BY-NC-SA 4.0; see NOTICE.
-->

# tk-train

Training half of the 🤗 Tokenizers library.

This crate builds on top of [`tk_encode`] and provides everything related to
*training* a tokenizer: the [`Trainer`] trait, every concrete `*Trainer`, the
[`TrainerWrapper`] enum and the [`Trainable`] extension (the `get_trainer`
association that used to live on `tk_encode::Model`).

There is no tokenizer-level `train` / `train_from_files` entry point: it was an
extension trait on `tk_convert`'s `TokenizerImpl`, and nothing in `tk-encode`
replaces that type -- the pipeline tokenizer hands out its model by shared
reference only. Drive a trainer directly instead: `feed`, then `train`.

License: Apache-2.0 AND CC-BY-NC-SA-4.0
