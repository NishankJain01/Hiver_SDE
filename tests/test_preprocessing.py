"""
Unit Tests for Data Preprocessing and Normalization.
"""

import pytest
from src.data.preprocessor import clean_tweet_text, is_english_text, normalize_whitespace


def test_clean_tweet_text_removes_mentions_and_signatures():
    raw = "@AmazonHelp @115820 Where is my order package? https://t.co/xyz123 ^KM"
    cleaned = clean_tweet_text(raw, mask_urls=True)
    assert "@AmazonHelp" not in cleaned
    assert "@115820" not in cleaned
    assert "^KM" not in cleaned
    assert "[URL]" in cleaned
    assert "Where is my order package?" in cleaned


def test_clean_tweet_text_html_entities():
    raw = "Fish &amp; Chips order was delayed &lt; 3 days"
    cleaned = clean_tweet_text(raw)
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "<" in cleaned


def test_is_english_text_filters_languages():
    english_text = "My package has not arrived and tracking is stuck."
    japanese_text = "公式様から直々にお礼を言われるとは！ありがとうございます！"
    spanish_text = "Hola, por favor necesito ayuda con mi pedido que no ha llegado."
    french_text = "Bonjour, j'ai un problème avec ma commande s'il vous plaît."

    assert is_english_text(english_text) is True
    assert is_english_text(japanese_text) is False
    assert is_english_text(spanish_text) is False
    assert is_english_text(french_text) is False


def test_normalize_whitespace():
    text = "Hello    world  \n\t  from   Amazon Support   "
    normalized = normalize_whitespace(text)
    assert normalized == "Hello world from Amazon Support"
