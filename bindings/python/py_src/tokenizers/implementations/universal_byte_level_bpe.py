# Adapted by Samsung Electronics Co., Ltd. in 2026 from byte_level_bpe.py of Hugging Face Tokenizers
# (Apache-2.0). Samsung's changes are licensed under CC BY-NC-SA 4.0; see NOTICE.

from typing import Dict, Iterator, List, Optional, Tuple, Union

from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, Sequence, unicode_normalizer_from_str

from .base_tokenizer import BaseTokenizer


class UniversalByteLevelBPETokenizer(BaseTokenizer):
    """UniversalByteLevelBPETokenizer

    Represents a byte-level BPE over the Universal Byte-Level Encoding (UBE): each character is
    written with its UTF-8 bytes when it has at most two of them, and with the bytes of its
    UTF-16LE code units otherwise, using a 512-symbol base alphabet.
    """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        merges: Optional[Union[str, List[Tuple[str, str]]]] = None,
        add_prefix_space: bool = False,
        lowercase: bool = False,
        dropout: Optional[float] = None,
        unicode_normalizer: Optional[str] = None,
        continuing_subword_prefix: Optional[str] = None,
        end_of_word_suffix: Optional[str] = None,
        trim_offsets: bool = False,
        use_regex: bool = True,
        max_token_length: Optional[int] = None,
    ):
        if vocab is not None and merges is not None:
            tokenizer = Tokenizer(
                BPE(
                    vocab,
                    merges,
                    dropout=dropout,
                    continuing_subword_prefix=continuing_subword_prefix or "",
                    end_of_word_suffix=end_of_word_suffix or "",
                )
            )
        else:
            tokenizer = Tokenizer(BPE(dropout=dropout))

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = pre_tokenizers.UniversalByteLevel(
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            use_regex=use_regex,
        )
        tokenizer.decoder = decoders.UniversalByteLevel()
        post_processor = processors.UniversalByteLevel(trim_offsets=trim_offsets)
        if trim_offsets:
            post_processor.add_prefix_space = add_prefix_space
        tokenizer.post_processor = post_processor

        parameters = {
            "model": "UniversalByteLevelBPE",
            "add_prefix_space": add_prefix_space,
            "lowercase": lowercase,
            "dropout": dropout,
            "unicode_normalizer": unicode_normalizer,
            "continuing_subword_prefix": continuing_subword_prefix,
            "end_of_word_suffix": end_of_word_suffix,
            "trim_offsets": trim_offsets,
            "use_regex": use_regex,
            "max_token_length": max_token_length,
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(vocab_filename: str, merges_filename: str, **kwargs):
        vocab, merges = BPE.read_file(vocab_filename, merges_filename)
        return UniversalByteLevelBPETokenizer(vocab, merges, **kwargs)

    def _trainer(
        self,
        vocab_size: int,
        min_frequency: int,
        show_progress: bool,
        special_tokens: List[Union[str, AddedToken]],
        max_token_length: Optional[int],
    ):
        if max_token_length is None:
            max_token_length = self._parameters.get("max_token_length")
        return trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.UniversalByteLevel.alphabet(),
            max_token_length=max_token_length,
        )

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
        max_token_length: Optional[int] = None,
    ):
        """Train the model using the given files"""

        trainer = self._trainer(vocab_size, min_frequency, show_progress, special_tokens, max_token_length)
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
        length: Optional[int] = None,
        max_token_length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        trainer = self._trainer(vocab_size, min_frequency, show_progress, special_tokens, max_token_length)
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )
