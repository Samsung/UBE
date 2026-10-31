// Copyright 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: CC-BY-NC-SA-4.0

//! Train a universal byte-level (UBE) BPE tokenizer and write it as a `tokenizer.json`.
//!
//! UBE spells a character of one or two UTF-8 bytes with those bytes, in the GPT-2 byte-level
//! alphabet, and a longer character with the bytes of its UTF-16LE code units, at
//! `U+E100 + byte`: a base alphabet of 512 symbols. Training is BPE over that spelling. Every line
//! of the input is split with the GPT-2 pattern, every piece is spelled with
//! `tk_encode::utils::universal_byte_level::spell`, and the trainer starts from all 512 symbols.
//! The model is written with `"universal_byte_level": true`, so the file loads, encodes and
//! decodes like any other `tokenizer.json`.
//!
//! ```text
//! cargo run --release --example train_universal_byte_level -- \
//!     --vocab-size 32000 --special-token "<s>" --special-token "</s>" \
//!     --output tokenizer.json corpus.txt [more.txt ...]
//! ```
//!
//! Every line is one sequence, line ending included. Options: `--vocab-size` (default 30000),
//! `--min-frequency` (default 2), `--special-token` (repeatable, at least one character per value),
//! `--max-token-length`, and `--output` (default `tokenizer.json`).
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use tk_encode::models::bpe::{BpeConfig, PipelineBPE};
use tk_encode::pipeline::{
    PipelineModel, PipelineNormalizer, PipelinePostProcessor, PipelinePreTokenizer,
    PipelineTokenizer, PreTokenizer, PreTokenizerScratch,
};
use tk_encode::pre_tokenizers::split::{Split, SplitPattern};
use tk_encode::tokenizer::SplitDelimiterBehavior;
use tk_encode::utils::byte_level::GPT2_REGEX_STR;
use tk_encode::utils::universal_byte_level::{alphabet, spell};
use tk_encode::vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary};
use tk_train::{BpeTrainer, Trainer};

const USAGE: &str = "usage: train_universal_byte_level [--vocab-size N] [--min-frequency N] \
                     [--special-token TOKEN]... [--max-token-length N] [--output PATH] FILE...";

struct Args {
    files: Vec<PathBuf>,
    output: PathBuf,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
    max_token_length: Option<usize>,
}

fn parse_args(mut argv: impl Iterator<Item = String>) -> Result<Args, String> {
    fn number<T: FromStr>(option: &str, value: Option<String>) -> Result<T, String> {
        let value = value.ok_or_else(|| format!("{option} needs a value"))?;
        value
            .parse()
            .map_err(|_| format!("{option}: `{value}` is not a number"))
    }

    let mut args = Args {
        files: Vec::new(),
        output: PathBuf::from("tokenizer.json"),
        vocab_size: 30000,
        min_frequency: 2,
        special_tokens: Vec::new(),
        max_token_length: None,
    };
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--vocab-size" => args.vocab_size = number(&arg, argv.next())?,
            "--min-frequency" => args.min_frequency = number(&arg, argv.next())?,
            "--max-token-length" => args.max_token_length = Some(number(&arg, argv.next())?),
            "--special-token" => {
                let token = argv.next().ok_or("--special-token needs a value")?;
                if token.is_empty() {
                    return Err("--special-token needs at least one character".into());
                }
                args.special_tokens.push(token);
            }
            "--output" => args.output = argv.next().ok_or("--output needs a value")?.into(),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            option if option.starts_with("--") => return Err(format!("unknown option {option}")),
            _ => args.files.push(arg.into()),
        }
    }
    if args.files.is_empty() {
        return Err("no input files".into());
    }
    Ok(args)
}

/// Every line of every file, in order and with its line ending. The first read error ends the
/// stream and is kept in `error`.
struct Lines {
    files: std::vec::IntoIter<PathBuf>,
    current: Option<(PathBuf, BufReader<File>)>,
    error: Arc<Mutex<Option<String>>>,
}

impl Lines {
    fn new(files: Vec<PathBuf>, error: Arc<Mutex<Option<String>>>) -> Self {
        Self {
            files: files.into_iter(),
            current: None,
            error,
        }
    }

    fn fail(&mut self, path: &std::path::Path, e: std::io::Error) -> Option<String> {
        *self.error.lock().unwrap() = Some(format!("{}: {e}", path.display()));
        self.files = Vec::new().into_iter();
        self.current = None;
        None
    }
}

impl Iterator for Lines {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            if self.current.is_none() {
                let path = self.files.next()?;
                match File::open(&path) {
                    Ok(file) => {
                        self.current = Some((path, BufReader::with_capacity(1 << 20, file)));
                    }
                    Err(e) => return self.fail(&path, e),
                }
            }
            let (path, reader) = self.current.as_mut().expect("a file is open");
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => self.current = None,
                Ok(_) => return Some(line),
                Err(e) => {
                    let path = path.clone();
                    return self.fail(&path, e);
                }
            }
        }
    }
}

fn main() {
    let args = parse_args(std::env::args().skip(1)).unwrap_or_else(|e| {
        eprintln!("{e}\n{USAGE}");
        std::process::exit(2);
    });
    if let Err(e) = train(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn train(args: Args) -> tk_encode::Result<()> {
    // The GPT-2 split, as the reader builds it for a `Split` on that pattern.
    let split = Split::native(
        SplitPattern::Regex(GPT2_REGEX_STR.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )?
    .canonicalized_for_pipeline()?;
    let pre_tokenizer = PipelinePreTokenizer::Split(split);
    let process = |sequence: &str| -> tk_encode::Result<Vec<String>> {
        let mut spans = Vec::new();
        pre_tokenizer.pre_tokenize(sequence, &mut PreTokenizerScratch::default(), &mut spans)?;
        Ok(spans
            .iter()
            .map(|span| spell(&sequence[span.range()]))
            .collect())
    };

    let special_tokens: Vec<AddedToken> = args
        .special_tokens
        .iter()
        .map(|token| AddedToken::from(token.as_str(), true))
        .collect();
    let mut trainer = BpeTrainer::builder()
        .vocab_size(args.vocab_size)
        .min_frequency(args.min_frequency)
        .special_tokens(special_tokens)
        .initial_alphabet(alphabet().collect::<HashSet<char>>())
        .max_token_length(args.max_token_length)
        .build();

    let error = Arc::new(Mutex::new(None));
    trainer.feed(Lines::new(args.files, Arc::clone(&error)), process)?;
    if let Some(e) = error.lock().unwrap().take() {
        return Err(e.into());
    }
    let (vocab, merges, special_tokens) = trainer.train_vocab()?;
    let merge_count = merges.len();

    let model = PipelineBPE::from_config(BpeConfig {
        vocab: vocab.clone(),
        merges,
        universal_byte_level: true,
        ..BpeConfig::default()
    })?;
    let mut added_vocabulary = AddedVocabulary::new();
    added_vocabulary.add_special_tokens(
        special_tokens,
        vocab.len(),
        |token| vocab.get(token).copied(),
        None::<&PipelineNormalizer>,
    )?;
    let tokenizer = PipelineTokenizer::from_parts(
        added_vocabulary,
        Vec::new(),
        pre_tokenizer,
        PipelineModel::BPE(model),
        PipelinePostProcessor::default(),
        None,
        Default::default(),
        None,
        None,
    );
    std::fs::write(&args.output, tk_serialize::to_json(&tokenizer)?)?;
    println!(
        "{}: {} tokens, {merge_count} merges",
        args.output.display(),
        vocab.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_tokens_have_content() {
        let args = ["--special-token", "", "corpus.txt"].map(str::to_owned);
        assert_eq!(
            parse_args(args.into_iter()).err().unwrap(),
            "--special-token needs at least one character"
        );
    }

    #[test]
    fn lines_keep_endings_and_file_order() {
        let dir = tempfile::tempdir().unwrap();
        let contents: &[&[u8]] = &[b"", b"a\n\r\nb\r\nlast", b"", "다음\n".as_bytes(), b""];
        let files = contents
            .iter()
            .enumerate()
            .map(|(i, bytes)| {
                let path = dir.path().join(format!("{i}.txt"));
                std::fs::write(&path, bytes).unwrap();
                path
            })
            .collect();
        let error = Arc::new(Mutex::new(None));
        let lines: Vec<_> = Lines::new(files, Arc::clone(&error)).collect();
        assert_eq!(lines, ["a\n", "\r\n", "b\r\n", "last", "다음\n"]);
        assert!(error.lock().unwrap().is_none());
    }

    #[test]
    fn the_first_read_error_ends_the_stream() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.txt");
        std::fs::write(&path, b"valid\n\xff\nlater\n").unwrap();
        let error = Arc::new(Mutex::new(None));
        let mut lines = Lines::new(
            vec![path.clone(), dir.path().join("missing.txt")],
            Arc::clone(&error),
        );
        assert_eq!(lines.next().as_deref(), Some("valid\n"));
        assert_eq!(lines.next(), None);
        let first_error = error.lock().unwrap().clone().unwrap();
        assert!(first_error.starts_with(&format!("{}:", path.display())));
        assert_eq!(lines.next(), None);
        assert_eq!(error.lock().unwrap().as_ref(), Some(&first_error));
    }

    #[test]
    fn file_errors_preserve_the_output() {
        let dir = tempfile::tempdir().unwrap();
        let valid = dir.path().join("valid.txt");
        let invalid = dir.path().join("invalid.txt");
        let output = dir.path().join("tokenizer.json");
        std::fs::write(&valid, "Hi 빙고 🩷💚\n").unwrap();
        std::fs::write(&invalid, b"valid\n\xff\n").unwrap();
        for path in [dir.path().join("missing.txt"), invalid] {
            for previous in [None, Some("previous tokenizer")] {
                if let Some(text) = previous {
                    std::fs::write(&output, text).unwrap();
                } else if output.exists() {
                    std::fs::remove_file(&output).unwrap();
                }
                let error = train(Args {
                    files: vec![valid.clone(), path.clone()],
                    output: output.clone(),
                    vocab_size: 520,
                    min_frequency: 1,
                    special_tokens: vec!["<s>".into()],
                    max_token_length: None,
                })
                .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .starts_with(&format!("{}:", path.display()))
                );
                assert_eq!(std::fs::read_to_string(&output).ok().as_deref(), previous);
            }
        }
    }
}
