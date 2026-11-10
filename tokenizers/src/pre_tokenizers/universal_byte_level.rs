// Adapted by Samsung Electronics Co., Ltd. in 2026 from pre_tokenizers/byte_level.rs
// of Hugging Face Tokenizers (Apache-2.0). Samsung's modifications and additions are
// Copyright 2026 Samsung Electronics Co., Ltd. and licensed under CC BY-NC-SA 4.0; see NOTICE.
// The GPT-2 splitting pattern originates from:
// https://github.com/openai/gpt-2/blob/master/src/encoder.py#L98
// SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-SA-4.0

//! Universal Byte-Level Encoding (UBE).
//!
//! UBE writes each character with the bytes of one of two encodings. A character whose UTF-8
//! form has one or two bytes contributes those bytes, written in the same visible alphabet as
//! [`ByteLevel`](super::byte_level::ByteLevel). A character of three or four UTF-8 bytes
//! contributes the bytes of its UTF-16LE code units instead (two bytes, or four for a surrogate
//! pair), written as `U+E100 + byte` in the Private Use Area:
//!
//! ```text
//!   "A"   U+0041    UTF-8     41           ->  "A"
//!   "é"   U+00E9    UTF-8     C3 A9        ->  "Ã" "©"
//!   "中"  U+4E2D    UTF-16LE  2D 4E        ->  U+E12D U+E14E
//!   "😀"  U+1F600   UTF-16LE  3D D8 00 DE  ->  U+E13D U+E1D8 U+E100 U+E1DE
//! ```
//!
//! The two halves form a 512-symbol base alphabet. The encoding of every byte is given by the half
//! its symbol belongs to, so no marker symbol is needed, and a merged token may mix both halves.
use ahash::{AHashMap, AHashSet};
use std::sync::LazyLock;

use crate::pre_tokenizers::byte_level::{bytes_char, process_offsets};
use crate::tokenizer::{
    Decoder, Encoding, PostProcessor, PreTokenizedString, PreTokenizer, Result,
    SplitDelimiterBehavior,
};
use crate::utils::macro_rules_attribute;
use crate::utils::SysRegex;
use serde::{Deserialize, Serialize};

/// `U+E100 + b` is the symbol of byte `b` of a UTF-16LE code unit.
const UTF16_BASE: u32 = 0xE100;

/// The same GPT-2 splitting pattern as the `ByteLevel` pre-tokenizer.
static RE: LazyLock<SysRegex> = LazyLock::new(|| {
    SysRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap()
});
static UTF8_CHAR: LazyLock<AHashMap<u8, char>> = LazyLock::new(bytes_char);
static CHAR_UTF8: LazyLock<AHashMap<char, u8>> =
    LazyLock::new(|| bytes_char().into_iter().map(|(b, c)| (c, b)).collect());

/// The symbol of byte `byte` of a UTF-16LE code unit.
fn utf16_char(byte: u8) -> char {
    // U+E100..=U+E1FF lies in the Private Use Area, so every value is a valid scalar value.
    char::from_u32(UTF16_BASE + u32::from(byte)).expect("U+E100..=U+E1FF are scalar values")
}

/// The UTF-16LE byte that `c` stands for, if `c` belongs to the UTF-16 half.
fn char_utf16(c: char) -> Option<u8> {
    let offset = (c as u32).wrapping_sub(UTF16_BASE);
    (offset < 256).then_some(offset as u8)
}

/// Whether `c` is written with the bytes of its UTF-16LE code units rather than its UTF-8 bytes.
fn uses_utf16(c: char) -> bool {
    c.len_utf8() > 2
}

fn default_true() -> bool {
    true
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// Pre-tokenizer, decoder and post-processor for Universal Byte-Level Encoding.
///
/// Its fields have the same meaning as in [`ByteLevel`](super::byte_level::ByteLevel).
#[macro_rules_attribute(impl_serde_type!)]
#[non_exhaustive]
pub struct UniversalByteLevel {
    /// Whether to add a leading space to the first word.
    pub add_prefix_space: bool,
    /// Whether the post processing step should trim offsets to avoid including whitespaces.
    pub trim_offsets: bool,
    /// Whether to use the standard GPT2 regex for whitespace splitting.
    #[serde(default = "default_true")]
    pub use_regex: bool,
}

impl Default for UniversalByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    }
}

impl UniversalByteLevel {
    pub fn new(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> Self {
        Self {
            add_prefix_space,
            trim_offsets,
            use_regex,
        }
    }

    /// The 512 base symbols: the 256 of the UTF-8 half and the 256 of the UTF-16 half.
    pub fn alphabet() -> AHashSet<char> {
        UTF8_CHAR
            .values()
            .copied()
            .chain((0..=255u8).map(utf16_char))
            .collect()
    }

    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    #[must_use]
    pub fn use_regex(mut self, v: bool) -> Self {
        self.use_regex = v;
        self
    }
}

/// As a `PreTokenizer`, `UniversalByteLevel` splits the input like `ByteLevel` does, then writes
/// every character with the base symbols of its encoding.
impl PreTokenizer for UniversalByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let re_ref: &SysRegex = &RE;
        pretokenized.split(|_, mut normalized| {
            if self.add_prefix_space && !normalized.get().starts_with(' ') {
                normalized.prepend(" ");
            }
            if self.use_regex {
                normalized.split(re_ref, SplitDelimiterBehavior::Isolated)
            } else {
                Ok(vec![normalized])
            }
        })?;
        pretokenized.normalize(|normalized| {
            let s = normalized.get();
            let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
            for c in s.chars() {
                // The first symbol replaces the character; every following one is an insertion,
                // so all of them keep the offsets of the original character.
                if uses_utf16(c) {
                    let mut units = [0u16; 2];
                    let bytes = c
                        .encode_utf16(&mut units)
                        .iter()
                        .flat_map(|unit| unit.to_le_bytes());
                    for (i, byte) in bytes.enumerate() {
                        transformations.push((utf16_char(byte), isize::from(i > 0)));
                    }
                } else {
                    let mut buf = [0u8; 4];
                    for (i, byte) in c.encode_utf8(&mut buf).bytes().enumerate() {
                        transformations.push((UTF8_CHAR[&byte], isize::from(i > 0)));
                    }
                }
            }
            normalized.transform(transformations, 0);
            Ok(())
        })
    }
}

/// Turns tokens written with the UBE base symbols back into text.
///
/// Consecutive symbols of one half are collected and decoded together, so a character may be
/// split across tokens. A run ends when a symbol of the other half, or a character outside the
/// alphabet, arrives; such a character is copied as is. Bytes that do not form text are replaced
/// with U+FFFD: a UTF-8 run is decoded lossily, and a UTF-16 run per code unit, with one more
/// U+FFFD for an odd trailing byte.
pub(crate) fn decode_symbols(tokens: &[String]) -> String {
    let mut text = String::new();
    let mut utf8: Vec<u8> = Vec::new();
    let mut utf16: Vec<u8> = Vec::new();
    for token in tokens {
        for c in token.chars() {
            if let Some(&byte) = CHAR_UTF8.get(&c) {
                flush_utf16(&mut utf16, &mut text);
                utf8.push(byte);
            } else if let Some(byte) = char_utf16(c) {
                flush_utf8(&mut utf8, &mut text);
                utf16.push(byte);
            } else {
                flush_utf8(&mut utf8, &mut text);
                flush_utf16(&mut utf16, &mut text);
                text.push(c);
            }
        }
    }
    flush_utf8(&mut utf8, &mut text);
    flush_utf16(&mut utf16, &mut text);
    text
}

fn flush_utf8(bytes: &mut Vec<u8>, text: &mut String) {
    if !bytes.is_empty() {
        text.push_str(&String::from_utf8_lossy(bytes));
        bytes.clear();
    }
}

fn flush_utf16(bytes: &mut Vec<u8>, text: &mut String) {
    if bytes.is_empty() {
        return;
    }
    let units = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]));
    text.extend(char::decode_utf16(units).map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER)));
    if bytes.len() % 2 == 1 {
        text.push(char::REPLACEMENT_CHARACTER);
    }
    bytes.clear();
}

/// As a `Decoder`, `UniversalByteLevel` merges all the tokens and decodes them in one step, since
/// a single token may hold only part of a character.
impl Decoder for UniversalByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(vec![decode_symbols(&tokens)])
    }
}

/// As a `PostProcessor`, `UniversalByteLevel` trims the offsets if necessary, exactly like
/// `ByteLevel`, which writes the space with the same symbol `Ġ`.
impl PostProcessor for UniversalByteLevel {
    fn added_tokens(&self, _is_pair: bool) -> usize {
        0
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if self.trim_offsets {
            for encoding in encodings.iter_mut() {
                process_offsets(encoding, self.add_prefix_space);
                encoding
                    .get_overflowing_mut()
                    .iter_mut()
                    .for_each(|encoding| process_offsets(encoding, self.add_prefix_space));
            }
        }
        for (i, encoding) in encodings.iter_mut().enumerate() {
            encoding.set_sequence_id(i);
        }
        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{OffsetReferential, OffsetType};

    fn splits(pretok: &UniversalByteLevel, input: &str) -> Vec<(String, (usize, usize))> {
        let mut pretokenized = PreTokenizedString::from(input);
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s.to_owned(), o))
            .collect()
    }

    #[test]
    fn alphabet_has_512_symbols() {
        assert_eq!(UniversalByteLevel::alphabet().len(), 512);
    }

    #[test]
    fn routes_each_character_by_its_utf8_length() {
        let ube = UniversalByteLevel::default().add_prefix_space(false);
        assert_eq!(
            splits(&ube, "Hello 中 é😀"),
            vec![
                ("Hello".to_owned(), (0, 5)),
                ("Ġ\u{E12D}\u{E14E}".to_owned(), (5, 9)),
                ("ĠÃ©".to_owned(), (9, 12)),
                ("\u{E13D}\u{E1D8}\u{E100}\u{E1DE}".to_owned(), (12, 16)),
            ]
        );
    }

    #[test]
    fn symbols_keep_the_offsets_of_their_character() {
        let ube = UniversalByteLevel::default().add_prefix_space(false);
        let mut pretokenized = PreTokenizedString::from("A한B");
        ube.pre_tokenize(&mut pretokenized).unwrap();
        pretokenized
            .tokenize(|normalized| {
                Ok(normalized
                    .get()
                    .char_indices()
                    .enumerate()
                    .map(|(id, (start, c))| {
                        crate::Token::new(id as u32, c.to_string(), (start, start + c.len_utf8()))
                    })
                    .collect())
            })
            .unwrap();
        let encoding = pretokenized
            .into_encoding(None, 0, OffsetType::Char)
            .unwrap();
        assert_eq!(encoding.get_offsets(), &[(0, 1), (1, 2), (1, 2), (2, 3)]);
    }

    #[test]
    fn decoding_inverts_pre_tokenization_across_token_boundaries() {
        let ube = UniversalByteLevel::default().add_prefix_space(false);
        let text = "Hi, 빙고 🩷💚! héllo 😀 नमस्ते 한국어\n";
        let symbols: String = splits(&ube, text).into_iter().map(|(s, _)| s).collect();
        assert_eq!(ube.decode(vec![symbols.clone()]).unwrap(), text);
        // One token per symbol: a character split across tokens still decodes.
        let tokens: Vec<String> = symbols.chars().map(String::from).collect();
        assert_eq!(ube.decode(tokens).unwrap(), text);
    }

    #[test]
    fn incomplete_sequences_decode_to_replacement_characters() {
        let decode = |tokens: &[&str]| {
            decode_symbols(&tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>())
        };
        // a lone UTF-8 lead byte, then a UTF-16 run with an odd trailing byte
        assert_eq!(
            decode(&["Ã", "\u{E12D}\u{E14E}\u{E100}"]),
            "\u{FFFD}中\u{FFFD}"
        );
        // an unpaired high surrogate
        assert_eq!(decode(&["\u{E13D}\u{E1D8}A"]), "\u{FFFD}A");
        // a character outside the alphabet is copied and ends the current run
        assert_eq!(
            decode(&["\u{E12D}", "中", "\u{E14E}"]),
            "\u{FFFD}中\u{FFFD}"
        );
    }

    #[test]
    fn serialization_round_trips() {
        let ube = UniversalByteLevel::default().add_prefix_space(false);
        let json = serde_json::to_string(&ube).unwrap();
        assert_eq!(
            json,
            r#"{"type":"UniversalByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}"#
        );
        assert_eq!(
            serde_json::from_str::<UniversalByteLevel>(&json).unwrap(),
            ube
        );
    }
}
