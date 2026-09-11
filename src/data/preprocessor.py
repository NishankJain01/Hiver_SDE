"""
Text Preprocessor and Normalization Module.

Provides robust cleaning for customer support tweets:
- Removes @mentions and author tags
- Normalizes URLs and links
- Strips Twitter support agent signatures (e.g., ^KM, ^NK, (1/2))
- Filters non-English messages
- Handles Windows encoding anomalies and whitespace normalization
"""

import re
import html
from typing import Optional


# Regex patterns
RE_MENTION = re.compile(r"@[A-Za-z0-9_]+", re.UNICODE)
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.UNICODE)
RE_AGENT_SIGNATURE = re.compile(r"\^[A-Z]{1,3}\b|\b[A-Z]{2}\b$|\(\d/\d\)", re.UNICODE)
RE_MULTISPACE = re.compile(r"\s+", re.UNICODE)
RE_SPECIAL_CHARS = re.compile(r"[\r\n\t]+", re.UNICODE)

# Common non-English stop patterns or scripts
NON_LATIN_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u0600-\u06ff\u0400-\u04ff]")


def is_english_text(text: str) -> bool:
    """
    Heuristic check to determine if a message is primarily English text.
    Rejects CJK, Cyrillic, Arabic, and excessive accented non-English tokens.
    """
    if not text or not isinstance(text, str):
        return False

    # Discard non-Latin scripts (Japanese, Chinese, Russian, Arabic)
    if NON_LATIN_SCRIPT.search(text):
        return False

    # Check ASCII / Latin character ratio
    total_chars = len(text)
    if total_chars == 0:
        return False

    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if (ascii_chars / total_chars) < 0.80:
        return False

    # Common non-English support indicators in dataset
    non_english_keywords = [
        "bonjour", "merci", "s'il vous", "cordialement", "pour votre",
        "hola", "gracias", "por favor", "ayuda", "saludos", "disculpe",
        "danke", "bitte", "schönen", "guten tag", "obrigado", "olá",
        "dopo", "grazie", "prego"
    ]
    lower_text = text.lower()
    for kw in non_english_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", lower_text):
            return False

    return True


def clean_tweet_text(text: str, remove_urls: bool = False, mask_urls: bool = True) -> str:
    """
    Cleans raw customer/support tweet text.

    Args:
        text: Raw text string.
        remove_urls: Whether to strip URLs completely.
        mask_urls: Whether to replace URLs with [URL] token.

    Returns:
        Cleaned, normalized string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Decode HTML entities (&amp; -> &, &lt; -> <)
    cleaned = html.unescape(text)

    # Remove @mentions
    cleaned = RE_MENTION.sub("", cleaned)

    # Handle URLs
    if remove_urls:
        cleaned = RE_URL.sub("", cleaned)
    elif mask_urls:
        cleaned = RE_URL.sub("[URL]", cleaned)

    # Strip Twitter support agent signatures (e.g., ^KM, ^NK, (1/2))
    cleaned = RE_AGENT_SIGNATURE.sub("", cleaned)

    # Normalize whitespace and newlines
    cleaned = RE_SPECIAL_CHARS.sub(" ", cleaned)
    cleaned = RE_MULTISPACE.sub(" ", cleaned)

    return cleaned.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace to a single space."""
    if not text or not isinstance(text, str):
        return ""
    return RE_MULTISPACE.sub(" ", text).strip()
