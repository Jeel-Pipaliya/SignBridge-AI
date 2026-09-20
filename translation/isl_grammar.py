"""
SignBridge AI - Indian Sign Language (ISL) Grammar Correction Engine (Week 5)
Transforms raw ISL token sequences into natural, grammatically structured English sentences.

Key Capabilities:
  - Token normalization (case, pronouns, synonyms).
  - ISL word-order restructuring (SOV -> SVO, Topic-Comment, Question-Word-Final).
  - Auxiliary verb, preposition, and copula insertion ("am going to", "is", "will").
  - Deterministic, offline-first execution with rule attribution.
"""

import re
from typing import Dict, List, NamedTuple, Optional, Tuple, Union


class GrammarResult(NamedTuple):
    """Container for grammar correction and sentence reconstruction output."""
    raw_text: str
    corrected_text: str
    language: str
    confidence: float
    rules_applied: List[str]

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text,
            "language": self.language,
            "confidence": round(float(self.confidence), 4),
            "rules_applied": self.rules_applied,
        }


# Canonical token normalization table
TOKEN_NORMALIZATION: Dict[str, str] = {
    "ME": "I",
    "MY": "MY",
    "YOU": "YOU",
    "YOUR": "YOUR",
    "HE": "HE",
    "SHE": "SHE",
    "WE": "WE",
    "THEY": "THEY",
    "COLLEGE": "COLLEGE",
    "SCHOOL": "SCHOOL",
    "OFFICE": "OFFICE",
    "HOME": "HOME",
    "GO": "GO",
    "COME": "COME",
    "EAT": "EAT",
    "DRINK": "DRINK",
    "WATER": "WATER",
    "FOOD": "FOOD",
    "HELP": "HELP",
    "NAME": "NAME",
    "WHAT": "WHAT",
    "WHERE": "WHERE",
    "HOW": "HOW",
    "WHEN": "WHEN",
    "WHO": "WHO",
    "WHY": "WHY",
    "TOMORROW": "TOMORROW",
    "YESTERDAY": "YESTERDAY",
    "TODAY": "TODAY",
    "NOW": "NOW",
    "HELLO": "HELLO",
    "PLEASE": "PLEASE",
    "THANK": "THANK",
    "THANKS": "THANKS",
    "YES": "YES",
    "NO": "NO",
}


class ISLGrammarEngine:
    """
    Rule-based deterministic grammar reconstruction prototype for ISL.
    """

    def __init__(self):
        self.rules_applied: List[str] = []

    def normalize_tokens(self, tokens: Union[List[str], str]) -> List[str]:
        """Convert input tokens or string into canonical uppercase token list."""
        if isinstance(tokens, str):
            token_list = tokens.strip().split()
        else:
            token_list = [str(t).strip() for t in tokens if str(t).strip() and str(t).strip() != "..."]

        normalized = []
        for t in token_list:
            up_t = t.upper()
            canon = TOKEN_NORMALIZATION.get(up_t, up_t)
            normalized.append(canon)
        return normalized

    def process(self, tokens: Union[List[str], str]) -> GrammarResult:
        """
        Main grammar correction pipeline:
          Raw ISL Tokens -> Normalization -> Pattern Matching -> Sentence Synthesis
        """
        self.rules_applied = []
        norm_tokens = self.normalize_tokens(tokens)
        raw_text = " ".join(norm_tokens)

        if not norm_tokens:
            return GrammarResult(
                raw_text="",
                corrected_text="",
                language="en",
                confidence=1.0,
                rules_applied=["EMPTY_INPUT"],
            )

        token_str = " ".join(norm_tokens)

        # 1. Single Token Special Cases
        if len(norm_tokens) == 1:
            token = norm_tokens[0]
            if token == "HELLO":
                return GrammarResult(raw_text, "Hello!", "en", 0.98, ["SINGLE_GREETING"])
            elif token in ("THANK", "THANKS"):
                return GrammarResult(raw_text, "Thank you.", "en", 0.98, ["SINGLE_COURTESY"])
            elif token == "PLEASE":
                return GrammarResult(raw_text, "Please.", "en", 0.95, ["SINGLE_COURTESY"])
            elif token == "YES":
                return GrammarResult(raw_text, "Yes.", "en", 0.95, ["SINGLE_AFFIRMATION"])
            elif token == "NO":
                return GrammarResult(raw_text, "No.", "en", 0.95, ["SINGLE_NEGATION"])
            elif len(token) == 1 and token.isalnum():
                return GrammarResult(raw_text, f"{token}.", "en", 0.90, ["SINGLE_CHARACTER"])

        # 2. ISL Question Patterns (Question-Word-Final)
        # "YOU NAME WHAT" -> "What is your name?"
        if re.search(r"\b(YOU|YOUR)\s+NAME\s+WHAT\b", token_str):
            self.rules_applied.append("QUESTION_NAME_WHAT")
            return GrammarResult(raw_text, "What is your name?", "en", 0.96, self.rules_applied)

        # "WHERE YOU GO" or "YOU GO WHERE" -> "Where are you going?"
        if re.search(r"\bWHERE\s+YOU\s+GO\b|\bYOU\s+GO\s+WHERE\b|\bYOU\s+WHERE\s+GO\b", token_str):
            self.rules_applied.append("QUESTION_WHERE_GOING")
            return GrammarResult(raw_text, "Where are you going?", "en", 0.94, self.rules_applied)

        # "HELP ME YOU" or "YOU HELP ME" -> "Can you help me?"
        if re.search(r"\bYOU\s+HELP\s+I\b|\bYOU\s+HELP\s+ME\b|\bHELP\s+ME\s+YOU\b", token_str):
            self.rules_applied.append("REQUEST_HELP")
            return GrammarResult(raw_text, "Can you help me?", "en", 0.95, self.rules_applied)

        # "I HELP YOU" -> "I can help you."
        if re.search(r"\bI\s+HELP\s+YOU\b", token_str):
            self.rules_applied.append("OFFER_HELP")
            return GrammarResult(raw_text, "I can help you.", "en", 0.95, self.rules_applied)

        # 3. Future Tense ISL Patterns (Time-Subject-Object-Verb or Subject-Time-Object-Verb)
        # "TOMORROW COLLEGE GO" or "I TOMORROW COLLEGE GO" -> "I will go to college tomorrow."
        match_tomorrow = re.search(r"\b(?:(I|WE|YOU|HE|SHE)\s+)?TOMORROW\s+(COLLEGE|SCHOOL|OFFICE|HOME)\s+GO\b", token_str)
        if match_tomorrow:
            subj = match_tomorrow.group(1) or "I"
            dest = match_tomorrow.group(2).lower()
            self.rules_applied.append("TIME_FUTURE_GO")
            if dest == "home":
                return GrammarResult(raw_text, f"{subj} will go home tomorrow.", "en", 0.94, self.rules_applied)
            else:
                return GrammarResult(raw_text, f"{subj} will go to {dest} tomorrow.", "en", 0.94, self.rules_applied)

        # 4. Present Continuous SOV Patterns
        # "I COLLEGE GO" -> "I am going to college."
        # "I HOME GO" -> "I am going home."
        match_go = re.search(r"\b(I|WE|YOU|HE|SHE|THEY)\s+(COLLEGE|SCHOOL|OFFICE|HOME|WATER)\s+GO\b", token_str)
        if match_go:
            subj = match_go.group(1)
            dest = match_go.group(2).lower()
            self.rules_applied.append("SOV_MOTION_CONTINUOUS")

            copula = "am" if subj == "I" else ("is" if subj in ("HE", "SHE") else "are")
            if dest == "home":
                return GrammarResult(raw_text, f"{subj} {copula} going home.", "en", 0.95, self.rules_applied)
            else:
                return GrammarResult(raw_text, f"{subj} {copula} going to {dest}.", "en", 0.95, self.rules_applied)

        # 5. Food & Drink Needs
        # "I WATER DRINK" -> "I want to drink water."
        # "I FOOD EAT" -> "I want to eat food."
        match_need = re.search(r"\b(I|WE)\s+(WATER|FOOD)\s+(DRINK|EAT)\b", token_str)
        if match_need:
            subj = match_need.group(1)
            item = match_need.group(2).lower()
            verb = match_need.group(3).lower()
            self.rules_applied.append("SOV_INGESTION_NEED")
            return GrammarResult(raw_text, f"{subj} want to {verb} {item}.", "en", 0.93, self.rules_applied)

        # 6. Fallback Heuristic Sentence Reconstruction
        # If no specific template matches, assemble cleanly with appropriate casing & punctuation
        self.rules_applied.append("HEURISTIC_ASSEMBLY")
        clean_words = []
        for w in norm_tokens:
            if w == "I":
                clean_words.append("I")
            else:
                clean_words.append(w.lower())

        assembled = " ".join(clean_words)
        if assembled:
            assembled = assembled[0].upper() + assembled[1:]
            if not assembled.endswith((".", "?", "!")):
                assembled += "."

        return GrammarResult(
            raw_text=raw_text,
            corrected_text=assembled,
            language="en",
            confidence=0.75,
            rules_applied=self.rules_applied,
        )
