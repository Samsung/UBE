# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Decoding parts of Universal Byte-Level Encoding (UBE) token sequences.

``Tokenizer.decode`` of a whole sequence and the streams of ``decoders.DecodeStream`` decode UBE tokens from the
start of a sequence. ``UniversalByteLevelDecoding`` adds what decoding a part of a sequence needs:

- ``boundaries``: the token positions where a character ends, so that the tokens before and after decode
  independently;
- ``aligned_offset``: the latest such position at or before a given one, as the start of the token windows that
  incremental detokenizers decode, for example the first window a few tokens before the end of a prompt;
- ``decode_slice``: the text of token ids cut from a longer sequence at any position;
- ``token_texts``: the text accumulated through each token that ends at a character boundary.

The helpers read the base symbols of each token, so they also apply to ``ByteLevel`` tokenizers, whose 256
symbols are the UTF-8 half of the UBE alphabet.
"""

import codecs
from typing import Dict, List, Optional, Sequence, Tuple

from tokenizers import Tokenizer, decoders

__all__ = ["UniversalByteLevelDecoding"]

# U+E100 + b is the symbol of byte b of a UTF-16LE code unit.
_UTF16_BASE = 0xE100
_REPLACEMENT = "�"
# Code points below U+0800 have at most two UTF-8 bytes, so UBE writes them with their UTF-8 bytes.
_FIRST_UTF16_ROUTED = 0x0800

_UTF8, _UTF16, _OTHER = 0, 1, 2


def _byte_level_chars() -> Dict[str, int]:
    """The character that stands for each byte in the GPT-2 byte-level alphabet (``ByteLevel.alphabet()``)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\N{INVERTED EXCLAMATION MARK}"), ord("\N{NOT SIGN}") + 1))
        + list(range(ord("\N{REGISTERED SIGN}"), ord("\N{LATIN SMALL LETTER Y WITH DIAERESIS}") + 1))
    )
    chars, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            chars[chr(byte)] = byte
        else:
            chars[chr(256 + extra)] = byte
            extra += 1
    return chars


_UTF8_SYMBOLS = _byte_level_chars()


def _symbol(char: str) -> Tuple[int, int]:
    """The half of the alphabet that ``char`` belongs to and the byte it stands for."""
    byte = _UTF8_SYMBOLS.get(char)
    if byte is not None:
        return _UTF8, byte
    offset = ord(char) - _UTF16_BASE
    if 0 <= offset < 256:
        return _UTF16, offset
    return _OTHER, 0


class _DecoderState:
    """The bytes that ``decoders.UniversalByteLevel`` holds back after each symbol.

    UTF-8 bytes and UTF-16LE bytes collect in two buffers; a symbol of the other half, or any other character,
    flushes the current buffer. A position is a character boundary when neither buffer holds part of a character:
    no incomplete UTF-8 sequence, no first byte of a code unit and no high surrogate.
    """

    __slots__ = ("utf8", "low", "high")

    def __init__(self):
        self.utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        self.low: Optional[int] = None
        self.high = False

    def push(self, kind: int, byte: int):
        if kind == _UTF8:
            self.low, self.high = None, False
            self.utf8.decode(bytes((byte,)))
        elif kind == _UTF16:
            self.utf8.reset()
            if self.low is None:
                self.low = byte
            else:
                unit = self.low | (byte << 8)
                self.low = None
                self.high = 0xD800 <= unit <= 0xDBFF
        else:
            self.utf8.reset()
            self.low, self.high = None, False

    def at_boundary(self) -> bool:
        return self.low is None and not self.high and not self.utf8.getstate()[0]


class UniversalByteLevelDecoding:
    """Decoding helpers for parts of UBE token sequences.

    Args:
        tokenizer (:class:`~tokenizers.Tokenizer`):
            A tokenizer with the ``UniversalByteLevel`` (or ``ByteLevel``) decoder. With 🤗 Transformers, pass the
            ``backend_tokenizer`` of a fast tokenizer.

        skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
            Whether special tokens are removed before decoding, as in ``Tokenizer.decode``.
    """

    def __init__(self, tokenizer: Tokenizer, skip_special_tokens: bool = True):
        if not isinstance(tokenizer.decoder, (decoders.UniversalByteLevel, decoders.ByteLevel)):
            raise ValueError("tokenizer must use the UniversalByteLevel or ByteLevel decoder")
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self._byte_level = isinstance(tokenizer.decoder, decoders.ByteLevel)
        self._tokens: Dict[int, Optional[str]] = {}
        self._symbols: Dict[int, Tuple[Tuple[int, int], ...]] = {}

    def _token(self, token_id: int) -> Optional[str]:
        """The token string that ``Tokenizer.decode`` passes to the decoder, or None when it skips the id."""
        if token_id not in self._tokens:
            token = self.tokenizer.id_to_token(token_id)
            # Native decoding applies the added vocabulary's content-based special-token selection.
            if (
                token is not None
                and self.skip_special_tokens
                and not self.tokenizer.decode([token_id], skip_special_tokens=True)
            ):
                token = None
            self._tokens[token_id] = token
            if token and self._byte_level:
                # ByteLevel uses the whole token's UTF-8 bytes when any character is outside its alphabet.
                raw = (
                    [_UTF8_SYMBOLS[c] for c in token]
                    if all(c in _UTF8_SYMBOLS for c in token)
                    else token.encode("utf-8")
                )
                self._symbols[token_id] = tuple((_UTF8, byte) for byte in raw)
            else:
                self._symbols[token_id] = tuple(_symbol(c) for c in token) if token else ()
        return self._tokens[token_id]

    def _token_symbols(self, token_id: int) -> Tuple[Tuple[int, int], ...]:
        self._token(token_id)
        return self._symbols[token_id]

    def boundaries(self, ids: Sequence[int]) -> List[bool]:
        """Character boundaries between the tokens of a sequence that starts at a character boundary.

        Args:
            ids (:obj:`List[int]`):
                Token ids, from the start of a sequence or from a character boundary.

        Returns:
            :obj:`List[bool]`: ``len(ids) + 1`` flags. Flag ``i`` is :obj:`True` when the first ``i`` tokens leave
            no pending bytes or high surrogate. Every :obj:`True` flag guarantees
            ``decode(ids[:i]) + decode(ids[i:]) == decode(ids)`` for any continuation.
        """
        state = _DecoderState()
        flags = [True]
        for token_id in ids:
            for kind, byte in self._token_symbols(token_id):
                state.push(kind, byte)
            flags.append(state.at_boundary())
        return flags

    def aligned_offset(self, ids: Sequence[int], offset: int) -> int:
        """The latest character boundary at or before ``offset`` in ``ids``.

        Incremental detokenizers decode a window of recent tokens and keep the new text. Starting each window
        at a character boundary keeps the decoded text of the window exact, for example::

            prefix_offset = decoding.aligned_offset(prompt_ids, len(prompt_ids) - 5)

        Args:
            ids (:obj:`List[int]`):
                Token ids from the start of a sequence; only the clamped prefix ``ids[:offset]`` is read,
                in one linear scan of its symbols.

            offset (:obj:`int`):
                A token position; values outside ``[0, len(ids)]`` are clamped.

        Returns:
            :obj:`int`: A position at or before the clamped offset such that ``ids[:i]`` ends at a character boundary.
        """
        offset = max(0, min(int(offset), len(ids)))
        flags = self.boundaries(ids[:offset])
        while not flags[offset]:
            offset -= 1
        return offset

    def decode_slice(self, ids: Sequence[int], ends_at_boundary: bool = True) -> str:
        """Decode token ids cut from a longer sequence at any token position.

        The slice may start inside a character. Each code unit written with UTF-16LE bytes is a pair of symbols,
        and every complete character contributes whole pairs, so the number of UTF-16LE symbols before the first
        symbol of another kind shows whether the slice starts at the second byte of a code unit. That byte, and
        any other part of the cut character, decode to U+FFFD. For token ids of encoded text, the characters after it
        decode exactly when this first run is followed by another kind of symbol or the slice ends at a character
        boundary.

        Args:
            ids (:obj:`List[int]`):
                Token ids cut from a sequence.

            ends_at_boundary (:obj:`bool`, defaults to :obj:`True`):
                Whether the slice ends at a character boundary, as at the end of a text. When the slice
                ends elsewhere and holds only UTF-16LE symbols, the code units are checked instead: UBE writes
                no code unit below U+0800 and pairs every surrogate. Pairing from the first symbol is preferred
                when both readings satisfy these rules; otherwise the reading whose first inconsistent unit
                ends later is used.

        Returns:
            :obj:`str`: The decoded text.
        """
        tokens = [token for token in map(self._token, ids) if token is not None]
        run, closed = [], False
        for token_id in ids:
            for kind, byte in self._token_symbols(token_id):
                if kind != _UTF16:
                    closed = True
                    break
                run.append(byte)
            if closed:
                break
        if not run:
            starts_at_second_byte = False
        elif closed or ends_at_boundary:
            starts_at_second_byte = len(run) % 2 == 1
        else:
            starts_at_second_byte = _second_byte_by_code_units(run)
        if not starts_at_second_byte:
            return self.tokenizer.decoder.decode(tokens) if tokens else ""
        first = next(i for i, token in enumerate(tokens) if token)
        tokens[first] = tokens[first][1:]
        return _REPLACEMENT + self.tokenizer.decoder.decode(tokens)

    def token_texts(self, ids: Sequence[int]) -> List[str]:
        """The text accumulated through each token that ends at a character boundary.

        Tokens accumulate until one ends at a character boundary. That token receives the group's decoded text,
        and earlier tokens in the group receive empty strings. The final token receives any remaining text.
        The texts join to ``Tokenizer.decode(ids)``.

        Args:
            ids (:obj:`List[int]`):
                Token ids from the start of a sequence or from a character boundary.

        Returns:
            :obj:`List[str]`: One string per token.
        """
        flags = self.boundaries(ids)
        texts = [""] * len(ids)
        start = 0
        for end in range(1, len(ids) + 1):
            if flags[end] or end == len(ids):
                texts[end - 1] = self.tokenizer.decode(
                    list(ids[start:end]), skip_special_tokens=self.skip_special_tokens
                )
                start = end
        return texts


def _second_byte_by_code_units(run: List[int]) -> bool:
    """For bytes of UTF-16LE code units that may start at either byte of a unit, whether pairing them from the
    second byte is the reading that UBE can produce; the first reading is kept when both remain possible."""

    def first_impossible(units: List[int]) -> Optional[int]:
        high = False
        for i, unit in enumerate(units):
            if unit < _FIRST_UTF16_ROUTED:
                return i
            if 0xD800 <= unit <= 0xDBFF:
                if high:
                    return i
                high = True
            elif 0xDC00 <= unit <= 0xDFFF:
                if not high and i > 0:
                    return i
                high = False
            elif high:
                return i
        return None

    from_first = first_impossible([run[i] | (run[i + 1] << 8) for i in range(0, len(run) - 1, 2)])
    from_second = first_impossible([run[i] | (run[i + 1] << 8) for i in range(1, len(run) - 1, 2)])
    if from_first is None:
        return False
    if from_second is None:
        return True
    # Unit i of the first reading ends at byte 2i + 2, unit i of the second at byte 2i + 3.
    return 2 * from_first + 2 <= 2 * from_second + 3
