// Copyright 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: CC-BY-NC-SA-4.0

//! A universal byte-level (UBE) tokenizer trained with `BpeTrainer`: the GPT-2 split, every piece
//! spelled in the 512-symbol alphabet, and the result built as a `universal_byte_level` model.
//! `examples/train_universal_byte_level.rs` does the same from files.
use std::collections::HashSet;

use tk_encode::models::bpe::{BpeConfig, PipelineBPE};
use tk_encode::pipeline::{
    EncodeOptions, PipelineModel, PipelineNormalizer, PipelinePostProcessor, PipelinePreTokenizer,
    PipelineTokenizer, PreTokenizer, PreTokenizerScratch,
};
use tk_encode::pre_tokenizers::split::{Split, SplitPattern};
use tk_encode::tokenizer::SplitDelimiterBehavior;
use tk_encode::utils::byte_level::GPT2_REGEX_STR;
use tk_encode::utils::universal_byte_level::{alphabet, spell};
use tk_encode::vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary};
use tk_train::{BpeTrainer, Trainer};

const CORPUS: &[&str] = &[
    "Hello, world! Hello again, world.\n",
    "안녕하세요, 세계! 한국어 문장입니다. 안녕하세요!\n",
    "你好，世界！这是中文句子。你好！\n",
    "नमस्ते दुनिया, यह हिंदी वाक्य है। नमस्ते!\n",
    "Héllo wörld, ça va? ¿Qué tal? Héllo!\n",
    "Emoji 😀😀 and more 😀 emoji\n",
];

fn train(corpus: &[&str], vocab_size: usize) -> PipelineTokenizer {
    let pre_tokenizer = PipelinePreTokenizer::Split(
        Split::native(
            SplitPattern::Regex(GPT2_REGEX_STR.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .unwrap()
        .canonicalized_for_pipeline()
        .unwrap(),
    );
    let process = |sequence: &str| -> tk_encode::Result<Vec<String>> {
        let mut spans = Vec::new();
        pre_tokenizer.pre_tokenize(sequence, &mut PreTokenizerScratch::default(), &mut spans)?;
        Ok(spans
            .iter()
            .map(|span| spell(&sequence[span.range()]))
            .collect())
    };
    let mut trainer = BpeTrainer::builder()
        .show_progress(false)
        .vocab_size(vocab_size)
        .special_tokens(vec![AddedToken::from("<s>", true)])
        .initial_alphabet(alphabet().collect::<HashSet<char>>())
        .build();
    trainer.feed(corpus.iter(), process).unwrap();
    let (vocab, merges, special_tokens) = trainer.train_vocab().unwrap();

    let model = PipelineBPE::from_config(BpeConfig {
        vocab: vocab.clone(),
        merges,
        universal_byte_level: true,
        ..BpeConfig::default()
    })
    .unwrap();
    let mut added_vocabulary = AddedVocabulary::new();
    added_vocabulary
        .add_special_tokens(
            special_tokens,
            vocab.len(),
            |token| vocab.get(token).copied(),
            None::<&PipelineNormalizer>,
        )
        .unwrap();
    PipelineTokenizer::from_parts(
        added_vocabulary,
        Vec::new(),
        pre_tokenizer,
        PipelineModel::BPE(model),
        PipelinePostProcessor::default(),
        None,
        Default::default(),
        None,
        None,
    )
}

fn ids(tokenizer: &PipelineTokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, &EncodeOptions::default())
        .wait()
        .unwrap()
        .iter()
        .flat_map(|encoding| encoding.ids())
        .map(|token| token.id())
        .collect()
}

fn vocabulary(tokenizer: &PipelineTokenizer) -> HashSet<String> {
    let model = tokenizer.get_model();
    (0..).map_while(|id| model.id_to_token(id)).collect()
}

#[test]
fn the_vocabulary_holds_every_atom_and_learns_merges() {
    let tokenizer = train(CORPUS, 600);
    let vocabulary = vocabulary(&tokenizer);
    assert!(alphabet().all(|atom| vocabulary.contains(&atom.to_string())));
    assert!(vocabulary.contains("<s>"));
    assert!(vocabulary.len() > 513, "no merges were learned");
}

#[test]
fn before_any_merge_a_character_costs_its_routed_bytes() {
    // 512 atoms and one special token leave no room for a merge.
    let tokenizer = train(CORPUS, 513);
    assert_eq!(vocabulary(&tokenizer).len(), 513);
    for (text, atoms) in [
        ("A", 1),
        ("é", 2),
        ("中", 2),
        ("한", 2),
        ("न", 2),
        ("😀", 4),
    ] {
        assert_eq!(ids(&tokenizer, text).len(), atoms, "{text}");
    }
}

#[test]
fn text_round_trips_through_the_written_json() {
    let json = tk_serialize::to_json(&train(CORPUS, 600)).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["model"]["universal_byte_level"], true);

    let tokenizer = tk_serialize::from_json(&json).unwrap();
    let unseen = [
        "unseen 文字 and 😀 again",
        "a literal Ġ, Ċ and \u{E100}",
        "<s>special</s>",
        "",
    ];
    for text in CORPUS.iter().copied().chain(unseen) {
        let ids = ids(&tokenizer, text);
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text, "{text:?}");
    }
}

#[test]
fn frequent_pairs_keep_their_merge_order() {
    let trainer = BpeTrainer::builder()
        .show_progress(false)
        .vocab_size(514)
        .initial_alphabet(alphabet().collect::<HashSet<char>>())
        .build();
    let expected = vec![("a".into(), "b".into()), ("ab".into(), "c".into())];
    for count in [3, 3_000_000_000] {
        let words = [(spell("abc").into(), count)].into_iter().collect();
        let (vocab, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(vocab.len(), 514);
        assert_eq!(merges, expected);
    }
}
