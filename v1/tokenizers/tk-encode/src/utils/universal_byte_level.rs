// Copyright 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: CC-BY-NC-SA-4.0

//! The alphabet a universal byte-level (UBE) vocabulary is written in, and how it decodes.
//!
//! UBE is byte-level BPE that picks the encoding per character. A character of one or two UTF-8
//! bytes contributes those bytes, spelled in the GPT-2 byte-level alphabet exactly as `byte_level`
//! spells them. A character of three or four UTF-8 bytes contributes its UTF-16LE code units
//! instead -- two bytes, or four for a surrogate pair -- and byte `b` of those is spelled
//! `U+E100 + b`, in the Private Use Area. That is 512 atoms, and a merged token may mix both
//! halves:
//!
//! ```text
//!   "A"   U+0041   UTF-8  41           ->  "A"
//!   "é"   U+00E9   UTF-8  C3 A9        ->  "Ã" "©"
//!   "中"  U+4E2D   UTF-16LE  2D 4E     ->  U+E12D U+E14E
//!   "😀"  U+1F600  UTF-16LE  3D D8 00 DE  ->  U+E13D U+E1D8 U+E100 U+E1DE
//! ```
//!
//! This is why a UBE vocabulary is not turned into raw bytes the way a `byte_level` one is: the
//! same byte value means different things in the two halves, and a token can end inside a UTF-16
//! code unit, so no per-token byte string concatenates back into the text. The vocabulary stays as
//! written, text is converted to atoms one character at a time, and decoding is a two-buffer pass.
//!
//! Training needs the same spelling as text: [`spell`] writes a pre-token in the alphabet, and
//! [`alphabet`] is the base vocabulary a trainer starts from.
use crate::utils::byte_level::{BYTES_CHAR_LOOKUP, char_to_byte};

/// `U+E100 + b` spells byte `b` of a UTF-16LE code unit.
pub const UTF16_ATOM_BASE: u32 = 0xE100;

/// One atom of the alphabet: a byte, and which encoding it is a byte of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Atom {
    Utf8(u8),
    Utf16(u8),
}

/// The character that spells byte `byte` of a UTF-16LE code unit.
pub fn utf16_atom_char(byte: u8) -> char {
    // U+E100..=U+E1FF is in the Private Use Area, so every value here is a scalar value.
    char::from_u32(UTF16_ATOM_BASE + u32::from(byte)).expect("U+E100..=U+E1FF are scalar values")
}

/// The atom `c` spells, or `None` if `c` is not one of the 512.
#[inline]
pub fn char_to_atom(c: char) -> Option<Atom> {
    if let Some(byte) = char_to_byte(c) {
        return Some(Atom::Utf8(byte));
    }
    let offset = (c as u32).wrapping_sub(UTF16_ATOM_BASE);
    (offset < 256).then_some(Atom::Utf16(offset as u8))
}

/// Whether `c` is written as UTF-16LE code units rather than as UTF-8 bytes.
#[inline(always)]
pub fn routes_to_utf16(c: char) -> bool {
    c.len_utf8() > 2
}

/// The 512 characters of the alphabet: the 256 byte-level characters, which spell UTF-8 bytes,
/// then `U+E100..=U+E1FF`, which spell UTF-16LE bytes.
pub fn alphabet() -> impl Iterator<Item = char> {
    BYTES_CHAR_LOOKUP
        .iter()
        .copied()
        .chain((0..=255u8).map(utf16_atom_char))
}

/// `text` written in the alphabet, one character at a time: a character of one or two UTF-8
/// bytes as those bytes, a longer one as the bytes of its UTF-16LE code units.
pub fn spell(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for c in text.chars() {
        if routes_to_utf16(c) {
            for unit in c.encode_utf16(&mut [0; 2]) {
                out.extend(unit.to_le_bytes().map(utf16_atom_char));
            }
        } else {
            out.extend(
                c.encode_utf8(&mut [0; 4])
                    .bytes()
                    .map(|b| BYTES_CHAR_LOOKUP[b as usize]),
            );
        }
    }
    out
}

/// Turns a stream of alphabet characters back into text, with one buffer per half.
///
/// A run of one half is flushed when the other half starts, or when a character outside the
/// alphabet arrives, so at most one buffer is ever non-empty. Bytes that do not form text decode
/// to U+FFFD: lossily for UTF-8, and per code unit for UTF-16, where an odd trailing byte adds one
/// more.
pub(crate) struct UniversalDecoder {
    text: String,
    utf8: Vec<u8>,
    utf16: Vec<u8>,
}

impl UniversalDecoder {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            text: String::with_capacity(capacity),
            utf8: Vec::new(),
            utf16: Vec::new(),
        }
    }

    /// A token spelled in the alphabet. A character outside it is passed through as itself.
    pub(crate) fn push_token(&mut self, token: &str) {
        for c in token.chars() {
            match char_to_atom(c) {
                Some(Atom::Utf8(byte)) => {
                    self.flush_utf16();
                    self.utf8.push(byte);
                }
                Some(Atom::Utf16(byte)) => {
                    self.flush_utf8();
                    self.utf16.push(byte);
                }
                None => {
                    self.flush();
                    self.text.push(c);
                }
            }
        }
    }

    /// Text that is not spelled in the alphabet, such as an added token's content.
    pub(crate) fn push_text(&mut self, text: &str) {
        self.flush();
        self.text.push_str(text);
    }

    pub(crate) fn finish(mut self) -> String {
        self.flush();
        self.text
    }

    fn flush(&mut self) {
        self.flush_utf8();
        self.flush_utf16();
    }

    fn flush_utf8(&mut self) {
        if !self.utf8.is_empty() {
            self.text.push_str(&String::from_utf8_lossy(&self.utf8));
            self.utf8.clear();
        }
    }

    fn flush_utf16(&mut self) {
        if self.utf16.is_empty() {
            return;
        }
        let units = self
            .utf16
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]));
        self.text
            .extend(char::decode_utf16(units).map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER)));
        if self.utf16.len() % 2 == 1 {
            self.text.push(char::REPLACEMENT_CHARACTER);
        }
        self.utf16.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(tokens: &[&str]) -> String {
        let mut decoder = UniversalDecoder::with_capacity(0);
        for token in tokens {
            decoder.push_token(token);
        }
        decoder.finish()
    }

    #[test]
    fn the_two_halves_are_disjoint_and_cover_512_atoms() {
        let mut seen = std::collections::HashSet::new();
        for byte in 0..=255u8 {
            let utf8 = BYTES_CHAR_LOOKUP[byte as usize];
            let utf16 = utf16_atom_char(byte);
            assert_eq!(char_to_atom(utf8), Some(Atom::Utf8(byte)));
            assert_eq!(char_to_atom(utf16), Some(Atom::Utf16(byte)));
            seen.insert(utf8);
            seen.insert(utf16);
        }
        assert_eq!(seen.len(), 512);
        for c in ['\u{E0FF}', '\u{E200}', '中', ' '] {
            assert_eq!(char_to_atom(c), None, "{c:?}");
        }
    }

    #[test]
    fn the_alphabet_is_the_utf8_half_then_the_utf16_half() {
        let atoms: Vec<Atom> = alphabet().map(|c| char_to_atom(c).unwrap()).collect();
        let expected: Vec<Atom> = (0..=255u8)
            .map(Atom::Utf8)
            .chain((0..=255u8).map(Atom::Utf16))
            .collect();
        assert_eq!(atoms, expected);
    }

    #[test]
    fn spelling_stays_inside_the_alphabet() {
        let text = "Hi, 빙고 🩷💚! héllo 😀 नमस्ते 한국어 \u{E100}Ġ\n";
        assert!(spell(text).chars().all(|c| char_to_atom(c).is_some()));
    }

    #[test]
    fn routing_follows_the_utf8_length() {
        assert_eq!(spell("A é"), "AĠÃ©");
        assert_eq!(spell("中"), "\u{E12D}\u{E14E}");
        assert_eq!(spell("😀"), "\u{E13D}\u{E1D8}\u{E100}\u{E1DE}");
    }

    #[test]
    fn decoding_inverts_the_spelling_across_any_token_boundary() {
        let text = "Hi, 빙고 🩷💚! héllo 😀 नमस्ते 한국어\n";
        let spelled = spell(text);
        assert_eq!(decode(&[&spelled]), text);
        // Cut after every character: a boundary inside a code unit or a surrogate pair changes
        // nothing, because the buffers carry over.
        let atoms: Vec<String> = spelled.chars().map(String::from).collect();
        let atoms: Vec<&str> = atoms.iter().map(String::as_str).collect();
        assert_eq!(decode(&atoms), text);
    }

    #[test]
    fn bytes_that_are_not_text_decode_to_replacement_characters() {
        // a lone UTF-8 lead byte, then a UTF-16 run with an odd trailing byte
        assert_eq!(
            decode(&["Ã", "\u{E12D}\u{E14E}\u{E100}"]),
            "\u{FFFD}中\u{FFFD}"
        );
        // an unpaired high surrogate
        assert_eq!(decode(&["\u{E13D}\u{E1D8}A"]), "\u{FFFD}A");
        // a character outside the alphabet passes through and ends the run it interrupts
        assert_eq!(
            decode(&["\u{E12D}", "中", "\u{E14E}"]),
            "\u{FFFD}中\u{FFFD}"
        );
    }
}
