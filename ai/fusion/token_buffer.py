"""
SignBridge AI - Token Buffer & Sentence Manager (Week 5)
Handles:
  1. Token accumulation with confidence tracking and debounce cooldowns.
  2. Sentence finalization, backspace, and clearing operations.
  3. Duplicate suppression and punctuation handling.
"""

import time
from typing import List, Optional, Tuple, NamedTuple


class TokenEntry(NamedTuple):
    token: str
    confidence: float
    timestamp: float
    source: str  # "STATIC" or "DYNAMIC"


class TokenBuffer:
    """
    Robust sentence-level token accumulator with debouncing, undo/backspace,
    and sentence finalization.
    """

    def __init__(
        self,
        confirmation_frames: int = 5,
        debounce_seconds: float = 1.2,
    ):
        self.confirmation_frames = confirmation_frames
        self.debounce_seconds = debounce_seconds

        self.tokens: List[TokenEntry] = []
        self._candidate_token: Optional[str] = None
        self._candidate_count: int = 0
        self._candidate_conf: float = 0.0
        self._candidate_source: str = "STATIC"

        self._last_committed_token: Optional[str] = None
        self._last_committed_time: float = 0.0
        self.is_finalized: bool = False

    def update(
        self,
        token: str,
        confidence: float = 1.0,
        source: str = "STATIC",
    ) -> Optional[str]:
        """
        Ingest a detected sign and commit to token buffer if stability criteria are met.
        For DYNAMIC signs, confirmation can happen immediately if confidence is high.
        For STATIC signs, consecutive confirmation frames are required.

        Returns:
            The committed token string, or None if not committed.
        """
        if token in ("...", "", "None", None):
            self._candidate_token = None
            self._candidate_count = 0
            return None

        current_time = time.time()

        # Dynamic gestures are sequence-level events; they don't require multi-frame majority voting
        if source == "DYNAMIC":
            can_add = True
            if (
                token == self._last_committed_token
                and (current_time - self._last_committed_time) < self.debounce_seconds
            ):
                can_add = False

            if can_add:
                entry = TokenEntry(token=token, confidence=confidence, timestamp=current_time, source=source)
                self.tokens.append(entry)
                self._last_committed_token = token
                self._last_committed_time = current_time
                self.is_finalized = False
                return token
            return None

        # Static signs use consecutive frame confirmation
        if token == self._candidate_token:
            self._candidate_count += 1
            self._candidate_conf = max(self._candidate_conf, confidence)
        else:
            self._candidate_token = token
            self._candidate_count = 1
            self._candidate_conf = confidence
            self._candidate_source = source

        if self._candidate_count >= self.confirmation_frames:
            can_add = True
            if (
                token == self._last_committed_token
                and (current_time - self._last_committed_time) < self.debounce_seconds
            ):
                can_add = False

            if can_add:
                entry = TokenEntry(token=token, confidence=self._candidate_conf, timestamp=current_time, source=source)
                self.tokens.append(entry)
                self._last_committed_token = token
                self._last_committed_time = current_time
                self._candidate_count = 0
                self.is_finalized = False
                return token

        return None

    def add_space(self) -> None:
        """Add space separation."""
        if self.tokens and self.tokens[-1].token != " ":
            self.tokens.append(TokenEntry(token=" ", confidence=1.0, timestamp=time.time(), source="SYSTEM"))
            self._last_committed_token = None

    def backspace(self) -> Optional[TokenEntry]:
        """Delete last token in buffer."""
        if self.tokens:
            removed = self.tokens.pop()
            self._last_committed_token = None
            self.is_finalized = False
            return removed
        return None

    def clear(self) -> None:
        """Clear all tokens."""
        self.tokens.clear()
        self._candidate_token = None
        self._candidate_count = 0
        self._last_committed_token = None
        self.is_finalized = False

    def finalize_sentence(self) -> str:
        """Finalize and freeze the current sentence."""
        self.is_finalized = True
        return self.get_text()

    def get_tokens(self) -> List[str]:
        """Return raw list of token strings."""
        return [t.token for t in self.tokens if t.token != " "]

    def get_text(self) -> str:
        """Format token stream into a space-separated sentence."""
        words = []
        for t in self.tokens:
            if t.token == " ":
                continue
            words.append(t.token)
        return " ".join(words)
