"""
SignBridge AI - Grammar & Sentence Reconstruction Service
"""

import logging
from typing import List, Optional
from backend.app.schemas.recognition import GrammarCorrectResponse
from translation.isl_grammar import ISLGrammarEngine
from translation.translator import translate_sentence

logger = logging.getLogger("GrammarService")


class GrammarService:
    def __init__(self):
        self._grammar_engine = ISLGrammarEngine()

    def correct_and_translate(
        self,
        tokens: List[str],
        target_language: str = "en",
    ) -> GrammarCorrectResponse:
        """
        Reconstructs an ordered set of ISL tokens into a fluent English sentence,
        and optionally translates it to Hindi.
        """
        raw_tokens = [t.strip().upper() for t in tokens if t.strip() and t.strip() != "..."]
        raw_sentence = " ".join(raw_tokens)

        grammar_result = self._grammar_engine.process(raw_tokens)
        corrected_sentence = grammar_result.corrected_text or raw_sentence

        translated_sentence: Optional[str] = None
        if target_language.lower() in ("hi", "hindi"):
            translated_sentence = translate_sentence(corrected_sentence, target_language="hi")

        return GrammarCorrectResponse(
            raw_tokens=raw_tokens,
            raw_sentence=raw_sentence,
            corrected_sentence=corrected_sentence,
            language=target_language,
            translated_sentence=translated_sentence,
        )


grammar_service = GrammarService()
