// Copyright 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: CC-BY-NC-SA-4.0

use crate::decoders::DecoderWrapper;
use crate::pre_tokenizers::universal_byte_level::{self, decode_symbols};
use crate::tokenizer::{Decoder, Result};
use monostate::MustBe;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
/// Decoder for tokens written with the Universal Byte-Level Encoding base symbols, to be used with
/// the [`UniversalByteLevel`](crate::pre_tokenizers::universal_byte_level::UniversalByteLevel)
/// pre-tokenizer.
#[non_exhaustive]
pub struct UniversalByteLevel {
    #[serde(rename = "type")]
    type_: MustBe!("UniversalByteLevel"),
}

impl UniversalByteLevel {
    pub fn new() -> Self {
        Self {
            type_: MustBe!("UniversalByteLevel"),
        }
    }
}

impl Decoder for UniversalByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(vec![decode_symbols(&tokens)])
    }
}

/// As with `ByteLevel`, the pre-tokenizer can also be given as the decoder.
impl From<universal_byte_level::UniversalByteLevel> for DecoderWrapper {
    fn from(_: universal_byte_level::UniversalByteLevel) -> Self {
        DecoderWrapper::UniversalByteLevel(UniversalByteLevel::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_its_type_only() {
        let json = serde_json::to_string(&UniversalByteLevel::new()).unwrap();
        assert_eq!(json, r#"{"type":"UniversalByteLevel"}"#);
        serde_json::from_str::<UniversalByteLevel>(&json).unwrap();
    }

    #[test]
    fn the_pre_tokenizer_converts_to_this_decoder() {
        let decoder = DecoderWrapper::from(universal_byte_level::UniversalByteLevel::default());
        assert!(matches!(decoder, DecoderWrapper::UniversalByteLevel(_)));
        let json = serde_json::to_string(&decoder).unwrap();
        assert_eq!(json, r#"{"type":"UniversalByteLevel"}"#);
        let reloaded: DecoderWrapper = serde_json::from_str(&json).unwrap();
        assert!(matches!(reloaded, DecoderWrapper::UniversalByteLevel(_)));
    }

    #[test]
    fn decodes_across_tokens() {
        let tokens = [
            "Hi",
            "Ġ\u{E159}",
            "\u{E1BE}\u{E1E0}\u{E1AC}",
            "Ġ\u{E13E}\u{E1D8}",
            "\u{E177}\u{E1DE}\u{E13D}\u{E1D8}\u{E19A}\u{E1DC}",
        ];
        let decoded = UniversalByteLevel::new()
            .decode(tokens.iter().map(|t| t.to_string()).collect())
            .unwrap();
        assert_eq!(decoded, "Hi 빙고 🩷💚");
    }
}
