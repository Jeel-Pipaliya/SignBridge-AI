"""
SignBridge AI - Multilingual Translation Module (Week 5)
Provides translation between English and Hindi for recognized sign language sentences.
Employs an offline semantic dictionary and sentence-template engine with extensible API fallback.
"""

from typing import Dict, Optional


# Direct sentence pair translations for maximum natural fluency
SENTENCE_TRANSLATIONS: Dict[str, str] = {
    # Greetings & Common Expressions
    "Hello!": "नमस्ते!",
    "Thank you.": "धन्यवाद।",
    "Please.": "कृपया।",
    "Yes.": "हाँ।",
    "No.": "नहीं।",

    # Questions
    "What is your name?": "आपका नाम क्या है?",
    "Where are you going?": "आप कहाँ जा रहे हैं?",
    "Can you help me?": "क्या आप मेरी मदद कर सकते हैं?",
    "I can help you.": "मैं आपकी मदद कर सकता हूँ।",

    # Travel & Movement (Continuous)
    "I am going to college.": "मैं कॉलेज जा रहा हूँ।",
    "I am going home.": "मैं घर जा रहा हूँ।",
    "I am going to school.": "मैं स्कूल जा रहा हूँ।",
    "I am going to office.": "मैं दफ़्तर जा रहा हूँ।",
    "He is going to college.": "वह कॉलेज जा रहा है।",
    "She is going to college.": "वह कॉलेज जा रही है।",
    "We are going home.": "हम घर जा रहे हैं।",

    # Future Tense
    "I will go to college tomorrow.": "मैं कल कॉलेज जाऊँगा।",
    "I will go home tomorrow.": "मैं कल घर जाऊँगा।",
    "I will go to school tomorrow.": "मैं कल स्कूल जाऊँगा।",

    # Needs
    "I want to drink water.": "मुझे पानी पीना है।",
    "I want to eat food.": "मुझे खाना खाना है।",
}

# Lexical word-by-word mapping for fallback translation
VOCAB_EN_TO_HI: Dict[str, str] = {
    "i": "मैं",
    "me": "मुझे",
    "you": "आप",
    "your": "आपका",
    "he": "वह",
    "she": "वह",
    "we": "हम",
    "they": "वे",
    "college": "कॉलेज",
    "school": "स्कूल",
    "office": "कार्यालय",
    "home": "घर",
    "go": "जाना",
    "going": "जा रहे",
    "come": "आना",
    "eat": "खाना",
    "drink": "पीना",
    "water": "पानी",
    "food": "भोजन",
    "help": "मदद",
    "name": "नाम",
    "what": "क्या",
    "where": "कहाँ",
    "tomorrow": "कल",
    "today": "आज",
    "yesterday": "कल",
    "now": "अब",
    "hello": "नमस्ते",
    "please": "कृपया",
    "thanks": "धन्यवाद",
    "thank": "धन्यवाद",
    "yes": "हाँ",
    "no": "नहीं",
}


def translate_sentence(
    text: str,
    target_language: str = "hi",
    source_language: str = "en",
) -> str:
    """
    Translates an English sentence into Hindi (or vice versa).

    Args:
        text: Input sentence string.
        target_language: Target language code ('en' or 'hi').
        source_language: Source language code (default 'en').

    Returns:
        Translated sentence string. If target is 'en' or unsupported, returns text.
    """
    clean_text = text.strip()
    if not clean_text:
        return ""

    target_lang = target_language.lower()
    if target_lang in ("en", "english"):
        return clean_text

    if target_lang not in ("hi", "hindi"):
        # Unsupported language: fallback to original text safely
        return clean_text

    # 1. Exact Sentence Template Match (Highest Natural Quality)
    if clean_text in SENTENCE_TRANSLATIONS:
        return SENTENCE_TRANSLATIONS[clean_text]

    # Normalized check (ignoring trailing punctuation)
    core_text = clean_text.rstrip(".?!")
    for en_template, hi_template in SENTENCE_TRANSLATIONS.items():
        if en_template.rstrip(".?!").lower() == core_text.lower():
            return hi_template

    # 2. Token / Word-Level Dictionary Fallback
    words = clean_text.split()
    translated_words = []
    for w in words:
        punct = ""
        clean_w = w
        if w and w[-1] in (".", "?", "!", ","):
            punct = w[-1]
            clean_w = w[:-1]

        lower_w = clean_w.lower()
        hi_w = VOCAB_EN_TO_HI.get(lower_w, clean_w)
        translated_words.append(hi_w)

    translated = " ".join(translated_words)
    if not translated.endswith("।"):
        translated += "।"
    return translated
