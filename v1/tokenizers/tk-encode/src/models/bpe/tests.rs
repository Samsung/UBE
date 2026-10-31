// Modified by Samsung Electronics Co., Ltd. in 2026 for Universal Byte-Level Encoding (UBE).
// Samsung's modifications are licensed under CC BY-NC-SA 4.0; see NOTICE.

//! Tests for [`PipelineBPE`], the only BPE in this crate.
//!
//! Everything here goes through [`PipelineBPE::from_vocab_and_merges`], which is the serde-free door
//! both readers come in by. The config-shaped `BPE` — its builder, its serde, its `read_file` — and
//! the tests that pin the two engines against each other moved to `tk-convert` with the type.
use super::*;
use crate::pipeline;
use crate::tokenizer::Result;

use crate::{
    pipeline::Model as PipelineModel, utils::byte_level::BYTES_CHAR_LOOKUP,
    utils::universal_byte_level::utf16_atom_char,
};

const HELLO_VOCAB: &[(&str, u32)] = &[
    ("h", 0),
    ("e", 1),
    ("l", 2),
    ("o", 3),
    ("he", 4),
    ("hel", 5),
    ("hell", 6),
    ("hello", 7),
];
const HELLO_MERGES: &[(&str, &str)] = &[("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")];

fn v(pairs: &[(&str, u32)]) -> Vocab {
    pairs.iter().map(|&(s, i)| (s.into(), i)).collect()
}

fn m(pairs: &[(&str, &str)]) -> Merges {
    pairs.iter().map(|&(a, b)| (a.into(), b.into())).collect()
}

/// The `hello` model: `h`/`e`/`l`/`o` plus the four merges that fold them into one id.
fn hello(options: BpeConfig) -> Result<PipelineBPE> {
    PipelineBPE::from_config(BpeConfig {
        vocab: v(HELLO_VOCAB),
        merges: m(HELLO_MERGES),
        ..options
    })
}

fn pipeline_ids(model: &PipelineBPE, sequence: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut scratch = model.init_scratch();
    pipeline::Model::tokenize_pipeline(model, sequence, &mut scratch, &mut out).unwrap();
    out.iter().map(|t| t.id()).collect()
}

#[test]
fn applies_merges() {
    let pipeline = hello(BpeConfig::default()).unwrap();
    for (input, want) in [
        ("hello", vec![7]),
        ("hell", vec![6]),
        ("helo", vec![5, 3]),
        ("oleh", vec![3, 2, 1, 0]),
    ] {
        assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
    }
}

#[test]
fn empty_input_yields_no_tokens() {
    let pipeline = hello(BpeConfig::default()).unwrap();
    assert!(pipeline_ids(&pipeline, "").is_empty());
}

// The scratch pool hands the SAME scratch to successive encodes. A bug leaking
// state between calls (an undrained merge queue, a stale word buffer) would
// corrupt every encode after the first. Drive several inputs — including
// repeats and an empty string — through one reused scratch and check each still
// matches what a fresh scratch produces. This is the invariant the pool relies on.
#[test]
fn reused_scratch_matches_fresh() {
    let model = hello(BpeConfig::default()).unwrap();
    let mut scratch = model.init_scratch();
    for input in ["hello", "hell", "helo", "oleh", "hello", "", "hxe"] {
        let mut out = Vec::new();
        pipeline::Model::tokenize_pipeline(&model, input, &mut scratch, &mut out).unwrap();
        let got: Vec<u32> = out.iter().map(|t| t.id()).collect();
        assert_eq!(got, pipeline_ids(&model, input), "{input:?}");
    }
}

// A cache may forget a word, but it must never change one. Run every word twice
// through one scratch, the second time answered from the cache, against a model
// built with no cache at all.
#[test]
fn cached_ids_match_uncached() {
    let cached = hello(BpeConfig::default()).unwrap();
    let uncached = hello(BpeConfig {
        cache_capacity: 0,
        ..Default::default()
    })
    .unwrap();

    let mut scratch = cached.init_scratch();
    assert!(scratch.word_cache.is_some(), "nothing is being cached");
    for _ in 0..2 {
        for word in [
            "hello",
            "hell",
            "o",
            "hellohello",
            "hello-a-word-past-fifteen-bytes",
            "hxe",
        ] {
            let mut out = Vec::new();
            pipeline::Model::tokenize_pipeline(&cached, word, &mut scratch, &mut out).unwrap();
            let got: Vec<u32> = out.iter().map(|t| t.id()).collect();
            assert_eq!(got, pipeline_ids(&uncached, word), "{word:?}");
        }
    }
}

#[test]
fn unknown_char_without_unk_is_dropped() {
    let pipeline = hello(BpeConfig::default()).unwrap();
    // 'x' vanishes, making 'h' and 'e' adjacent, so the (h,e) merge applies.
    assert_eq!(pipeline_ids(&pipeline, "hxe"), vec![4]);
}

#[test]
fn unk_replaces_unknown_chars() {
    let mut vocab = v(HELLO_VOCAB);
    vocab.insert("<unk>".into(), 8);
    let pipeline = PipelineBPE::from_config(BpeConfig {
        vocab,
        merges: m(HELLO_MERGES),
        ..BpeConfig {
            unk_token: Some("<unk>".into()),
            ..Default::default()
        }
    })
    .unwrap();
    for (input, want) in [
        ("hxe", vec![0, 8, 1]),
        ("xh", vec![8, 0]),
        ("hxxe", vec![0, 8, 8, 1]),
        ("xx", vec![8, 8]),
    ] {
        assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
    }
}

#[test]
fn fused_unk_collapses_runs() {
    let mut vocab = v(HELLO_VOCAB);
    vocab.insert("<unk>".into(), 8);
    let pipeline = PipelineBPE::from_config(BpeConfig {
        vocab,
        merges: m(HELLO_MERGES),
        ..BpeConfig {
            unk_token: Some("<unk>".into()),
            fuse_unk: true,
            ..Default::default()
        }
    })
    .unwrap();
    for (input, want) in [
        ("hxxe", vec![0, 8, 1]),
        ("xxh", vec![8, 0]),
        ("xxxx", vec![8]),
        ("xhx", vec![8, 0, 8]),
    ] {
        assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
    }
}

fn byte_fallback_vocab() -> Vocab {
    let mut vocab = v(&[("h", 300), ("e", 301), ("<unk>", 400)]);
    vocab.extend((0..=255u8).map(|b| (format!("<0x{b:02X}>"), u32::from(b))));
    vocab
}

#[test]
fn byte_fallback_encodes_missing_chars_as_byte_tokens() {
    let pipeline = PipelineBPE::from_config(BpeConfig {
        vocab: byte_fallback_vocab(),
        merges: vec![],
        ..BpeConfig {
            byte_fallback: true,
            ..Default::default()
        }
    })
    .unwrap();
    // 'é' is not in the vocab: falls back to its UTF-8 bytes C3 A9
    assert_eq!(pipeline_ids(&pipeline, "hé"), vec![300, 0xC3, 0xA9]);
    assert_eq!(pipeline_ids(&pipeline, "🤗"), vec![0xF0, 0x9F, 0xA4, 0x97]);
    assert_eq!(pipeline_ids(&pipeline, "he"), vec![300, 301]);
}

#[test]
fn byte_fallback_wins_over_unk() {
    let pipeline = PipelineBPE::from_config(BpeConfig {
        vocab: byte_fallback_vocab(),
        merges: vec![],
        ..BpeConfig {
            byte_fallback: true,
            unk_token: Some("<unk>".into()),
            ..Default::default()
        }
    })
    .unwrap();
    assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
}

#[test]
fn ignore_merges_prefers_whole_word() {
    let pipeline = hello(BpeConfig {
        ignore_merges: true,
        ..Default::default()
    })
    .unwrap();
    // direct vocab hit bypasses the merge loop; a miss falls through to it
    assert_eq!(pipeline_ids(&pipeline, "hello"), vec![7]);
    assert_eq!(pipeline_ids(&pipeline, "helo"), vec![5, 3]);
}

#[test]
fn rejects_unsupported_configs() {
    // no merges: the merge-map derivation underflows on merges whose right token
    // is shorter than continuing_subword_prefix (pre-existing, unrelated)
    let build = |options: BpeConfig| {
        PipelineBPE::from_config(BpeConfig {
            vocab: v(HELLO_VOCAB),
            merges: vec![],
            ..options
        })
    };
    // dropout is a valid config this engine cannot run...
    assert!(
        build(BpeConfig {
            dropout: Some(0.5),
            ..Default::default()
        })
        .is_err()
    );
    // ...while a dropout outside 0..=1 was never a valid config at all. Both are rejected, but
    // the second is the range check that used to live in `BpeBuilder::build`.
    let out_of_range = build(BpeConfig {
        dropout: Some(1.5),
        ..Default::default()
    })
    .err()
    .unwrap();
    assert!(matches!(
        out_of_range.downcast_ref::<Error>(),
        Some(Error::InvalidDropout)
    ));
    // affixes are supported: `convert_affixed` decorates each character before the lookup
    assert!(
        build(BpeConfig {
            continuing_subword_prefix: Some("##".into()),
            ..Default::default()
        })
        .is_ok()
    );
    assert!(
        build(BpeConfig {
            end_of_word_suffix: Some("</w>".into()),
            ..Default::default()
        })
        .is_ok()
    );
    // no-op values must not be rejected: gpt2's tokenizer.json serializes
    // prefix/suffix as "" and the reference treats dropout 0.0 as disabled
    assert!(
        build(BpeConfig {
            continuing_subword_prefix: Some(String::new()),
            end_of_word_suffix: Some(String::new()),
            dropout: Some(0.0),
            ..Default::default()
        })
        .is_ok()
    );
}

#[test]
fn rejects_unk_token_missing_from_vocab() {
    assert!(
        hello(BpeConfig {
            unk_token: Some("<unk>".into()),
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn byte_fallback_with_missing_codes_errors() {
    // Incomplete <0xNN> coverage must be a build error, not a panic.
    assert!(
        hello(BpeConfig {
            byte_fallback: true,
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn rejects_merge_token_out_of_vocabulary() {
    // The merge-map derivation is now part of this constructor, so its errors are its own.
    let err = PipelineBPE::from_config(BpeConfig {
        vocab: v(HELLO_VOCAB),
        merges: m(&[("h", "z")]),
        ..BpeConfig::default()
    })
    .err()
    .unwrap();
    match err.downcast_ref::<Error>() {
        Some(Error::MergeTokenOutOfVocabulary(token)) => assert_eq!(token, "z"),
        other => panic!("expected MergeTokenOutOfVocabulary, got {other:?}"),
    }
}

fn projected(s: &str) -> String {
    s.bytes().map(|b| BYTES_CHAR_LOOKUP[b as usize]).collect()
}

/// A gpt2-shaped miniature: the 256 projected single-byte tokens
/// (id == byte value) plus `extra` tokens and merges, given in raw
/// space and projected here — like a real byte-level tokenizer.json,
/// whose vocab is stored in the projected alphabet.
fn byte_level_bpe(
    extra: &[(&str, u32)],
    merges: &[(&str, &str)],
    ignore_merges: bool,
) -> Result<PipelineBPE> {
    let mut vocab: Vocab = (0..=255u8)
        .map(|b| (BYTES_CHAR_LOOKUP[b as usize].to_string(), u32::from(b)))
        .collect();
    vocab.extend(extra.iter().map(|&(s, i)| (projected(s), i)));
    let merges: Merges = merges
        .iter()
        .map(|&(a, b)| (projected(a), projected(b)))
        .collect();
    PipelineBPE::from_config(BpeConfig {
        vocab,
        merges,
        ..BpeConfig {
            ignore_merges,
            byte_level: true,
            ..Default::default()
        }
    })
}

#[test]
fn byte_level_merges_raw_bytes() {
    let pipeline = byte_level_bpe(
        &[("he", 300), (" he", 301)],
        &[("h", "e"), (" ", "he")],
        false,
    )
    .unwrap();
    assert_eq!(pipeline_ids(&pipeline, " he"), vec![301]);
    // single bytes hit the un-projected single-byte tokens (id == byte value)
    assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
    assert_eq!(pipeline_ids(&pipeline, "\x00\x7f"), vec![0x00, 0x7f]);
    assert_eq!(
        pipeline_ids(&pipeline, "hé llo"),
        vec![
            u32::from(b'h'),
            0xC3,
            0xA9,
            u32::from(b' '),
            u32::from(b'l'),
            u32::from(b'l'),
            u32::from(b'o')
        ]
    );
}

#[test]
fn byte_level_ignore_merges_whole_word() {
    let pipeline = byte_level_bpe(&[(" hello", 300)], &[], true).unwrap();
    assert_eq!(pipeline_ids(&pipeline, " hello"), vec![300]);
    // not in vocab → falls through to single-byte atoms
    assert_eq!(
        pipeline_ids(&pipeline, "zz"),
        vec![u32::from(b'z'), u32::from(b'z')]
    );
}

#[test]
fn byte_level_id_to_token_spells_the_projected_alphabet() {
    let pipeline = byte_level_bpe(&[(" he", 300)], &[], true).unwrap();
    assert_eq!(pipeline.id_to_token(300).as_deref(), Some("Ġhe"));
    assert_eq!(pipeline.id_to_token(0xC3).as_deref(), Some("Ã"));
    assert_eq!(pipeline.id_to_token(301), None);
}

#[test]
fn byte_level_requires_full_byte_coverage() {
    // An ASCII-only vocab covers no control/high bytes: building the
    // byte-level pipeline must be a build error, not a panic.
    assert!(
        hello(BpeConfig {
            byte_level: true,
            ..Default::default()
        })
        .is_err()
    );
}

/// How a universal byte-level vocabulary spells `text`: the byte-level alphabet for a character of
/// one or two UTF-8 bytes, U+E100 + byte for the UTF-16LE code units of a longer one.
fn spelled(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        if c.len_utf8() <= 2 {
            out.push_str(&projected(c.encode_utf8(&mut [0; 4])));
        } else {
            for unit in c.encode_utf16(&mut [0; 2]).iter() {
                out.extend(unit.to_le_bytes().map(utf16_atom_char));
            }
        }
    }
    out
}

/// The 512 atoms -- UTF-8 byte `b` at id `b`, UTF-16 byte `b` at id `256 + b` -- plus `extra`
/// tokens and `merges`, both already spelled in the alphabet as a tokenizer.json stores them.
fn universal_bpe(
    extra: &[(&str, u32)],
    merges: &[(&str, &str)],
    options: BpeConfig,
) -> Result<PipelineBPE> {
    let mut vocab: Vocab = (0..=255u8)
        .flat_map(|b| {
            [
                (BYTES_CHAR_LOOKUP[b as usize].to_string(), u32::from(b)),
                (utf16_atom_char(b).to_string(), 256 + u32::from(b)),
            ]
        })
        .collect();
    vocab.extend(extra.iter().map(|&(s, i)| (s.to_string(), i)));
    PipelineBPE::from_config(BpeConfig {
        vocab,
        merges: m(merges),
        universal_byte_level: true,
        ..options
    })
}

/// `中` (two UTF-16 bytes) as one token, and `Ġ中` -- a space's UTF-8 atom merged with both of them.
fn universal_zh() -> PipelineBPE {
    universal_bpe(
        &[("\u{E12D}\u{E14E}", 600), ("Ġ\u{E12D}\u{E14E}", 601)],
        &[("\u{E12D}", "\u{E14E}"), ("Ġ", "\u{E12D}\u{E14E}")],
        BpeConfig::default(),
    )
    .unwrap()
}

#[test]
fn universal_byte_level_routes_each_character_by_its_utf8_length() {
    let pipeline = universal_zh();
    assert_eq!(spelled(" 中"), "Ġ\u{E12D}\u{E14E}");
    // the two halves merge into one token
    assert_eq!(pipeline_ids(&pipeline, " 中"), vec![601]);
    assert_eq!(pipeline_ids(&pipeline, "中"), vec![600]);
    // one or two UTF-8 bytes: those bytes
    assert_eq!(pipeline_ids(&pipeline, "Aé"), vec![0x41, 0xC3, 0xA9]);
    // four: the UTF-16LE surrogate pair D83D DE00
    assert_eq!(
        pipeline_ids(&pipeline, "😀"),
        vec![256 + 0x3D, 256 + 0xD8, 256, 256 + 0xDE]
    );
    // The input is text, not alphabet: a literal `Ġ` is U+0120, two UTF-8 bytes rather than the
    // space atom, and a literal U+E12D is a three-byte character whose code unit is E12D.
    assert_eq!(pipeline_ids(&pipeline, "Ġ"), vec![0xC4, 0xA0]);
    assert_eq!(
        pipeline_ids(&pipeline, "\u{E12D}"),
        vec![256 + 0x2D, 256 + 0xE1]
    );
}

#[test]
fn universal_byte_level_requires_all_512_atoms() {
    let pipeline = hello(BpeConfig {
        universal_byte_level: true,
        ..Default::default()
    });
    assert!(matches!(
        pipeline.err().unwrap().downcast_ref::<Error>(),
        Some(Error::UniversalAtomOutOfVocabulary(_))
    ));
    // the UTF-16 half is checked as well
    let mut vocab = universal_zh().to_config().unwrap().vocab;
    vocab.remove("\u{E1FF}");
    let err = PipelineBPE::from_config(BpeConfig {
        vocab,
        universal_byte_level: true,
        ..Default::default()
    })
    .err()
    .unwrap();
    match err.downcast_ref::<Error>() {
        Some(Error::UniversalAtomOutOfVocabulary(atom)) => assert_eq!(*atom, '\u{E1FF}'),
        other => panic!("expected UniversalAtomOutOfVocabulary, got {other:?}"),
    }
}

#[test]
fn universal_byte_level_refuses_options_that_look_raw_text_up() {
    // Each of these would look the input up in a vocabulary that is not spelled in it.
    let refused = [
        BpeConfig {
            byte_level: true,
            ..Default::default()
        },
        BpeConfig {
            ignore_merges: true,
            ..Default::default()
        },
        BpeConfig {
            continuing_subword_prefix: Some("##".into()),
            ..Default::default()
        },
        BpeConfig {
            end_of_word_suffix: Some("</w>".into()),
            ..Default::default()
        },
    ];
    for options in refused {
        assert!(universal_bpe(&[], &[], options).is_err());
    }
}

#[test]
fn universal_byte_level_keeps_its_alphabet_through_to_config() {
    let pipeline = universal_zh();
    assert!(pipeline.is_universal_byte_level() && !pipeline.is_byte_level());
    assert_eq!(
        pipeline.id_to_token(601).as_deref(),
        Some("Ġ\u{E12D}\u{E14E}")
    );
    assert_eq!(pipeline.id_to_token(0xC3).as_deref(), Some("Ã"));

    let config = pipeline.to_config().unwrap();
    assert!(config.universal_byte_level && !config.byte_level && !config.ignore_merges);
    assert_eq!(config.vocab.len(), 514);
    assert_eq!(config.vocab.get("Ġ\u{E12D}\u{E14E}"), Some(&601));
    let rebuilt = PipelineBPE::from_config(config).unwrap();
    for text in [" 中", "中 中", "Aé😀"] {
        assert_eq!(
            pipeline_ids(&rebuilt, text),
            pipeline_ids(&pipeline, text),
            "{text:?}"
        );
    }
}
