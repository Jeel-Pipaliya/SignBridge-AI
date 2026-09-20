"""
SignBridge AI - Unit Tests for ISL Grammar Correction & Multilingual Translation (Week 5)
Tests:
  - Canonical token normalization
  - Known ISL grammar patterns (SOV, Time-SOV, Question-Word-Final)
  - Single courtesies and greetings
  - Heuristic fallback assembly
  - English to Hindi sentence translation
  - Empty input handling
"""

import unittest
from translation.isl_grammar import ISLGrammarEngine, GrammarResult
from translation.translator import translate_sentence


class TestISLGrammarAndTranslation(unittest.TestCase):
    """Test suite for ISL grammar reconstruction and Hindi translation."""

    def setUp(self):
        self.engine = ISLGrammarEngine()

    def test_token_normalization(self):
        tokens = ["me", "college", "go"]
        norm = self.engine.normalize_tokens(tokens)
        self.assertEqual(norm, ["I", "COLLEGE", "GO"])

    def test_pattern_me_college_go(self):
        res = self.engine.process(["ME", "COLLEGE", "GO"])
        self.assertIsInstance(res, GrammarResult)
        self.assertEqual(res.corrected_text, "I am going to college.")
        self.assertIn("SOV_MOTION_CONTINUOUS", res.rules_applied)

    def test_pattern_you_name_what(self):
        res = self.engine.process(["YOU", "NAME", "WHAT"])
        self.assertEqual(res.corrected_text, "What is your name?")
        self.assertIn("QUESTION_NAME_WHAT", res.rules_applied)

    def test_pattern_i_home_go(self):
        res = self.engine.process(["I", "HOME", "GO"])
        self.assertEqual(res.corrected_text, "I am going home.")

    def test_pattern_tomorrow_college_go(self):
        res = self.engine.process(["TOMORROW", "COLLEGE", "GO"])
        self.assertEqual(res.corrected_text, "I will go to college tomorrow.")

    def test_pattern_single_greeting(self):
        res = self.engine.process(["HELLO"])
        self.assertEqual(res.corrected_text, "Hello!")

    def test_pattern_water_drink(self):
        res = self.engine.process(["I", "WATER", "DRINK"])
        self.assertEqual(res.corrected_text, "I want to drink water.")

    def test_empty_tokens(self):
        res = self.engine.process([])
        self.assertEqual(res.corrected_text, "")

    def test_translation_to_hindi(self):
        hi_greeting = translate_sentence("Hello!", target_language="hi")
        self.assertEqual(hi_greeting, "नमस्ते!")

        hi_name = translate_sentence("What is your name?", target_language="hi")
        self.assertEqual(hi_name, "आपका नाम क्या है?")

        hi_college = translate_sentence("I am going to college.", target_language="hi")
        self.assertEqual(hi_college, "मैं कॉलेज जा रहा हूँ।")

    def test_translation_english_noop(self):
        en_text = "I am going home."
        res = translate_sentence(en_text, target_language="en")
        self.assertEqual(res, en_text)


if __name__ == "__main__":
    unittest.main()
