"""
Text Utilities - Multi-language text processing
Handles word counting, text cleaning, and language-specific operations
"""
import re
from typing import Optional


def count_words(text: str, language: str = "English") -> int:
    """
    Count words in text, with language-specific handling.
    
    Args:
        text: The text to count words in
        language: The language of the text
    
    Returns:
        Word count (or character count for CJK languages)
    """
    if not text:
        return 0
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Language-specific word counting
    if language in ["Chinese", "Japanese"]:
        # For CJK languages, count characters (excluding whitespace and punctuation)
        # Remove whitespace
        text = re.sub(r'\s+', '', text)
        # Count characters (more meaningful than "words" for CJK)
        return len(text)
    
    elif language == "Korean":
        # Korean uses spaces, but also count characters for better accuracy
        # Use space-based counting but treat it as characters for consistency
        text = re.sub(r'\s+', '', text)
        return len(text)
    
    elif language in ["Thai", "Lao", "Khmer"]:
        # Thai doesn't use spaces between words
        # Count characters excluding whitespace
        text = re.sub(r'\s+', '', text)
        return len(text)
    
    else:
        # For space-separated languages (English, Spanish, French, etc.)
        # Split by whitespace and filter empty strings
        words = text.split()
        return len([w for w in words if w.strip()])


def get_reading_time(word_count: int, language: str = "English") -> int:
    """
    Calculate reading time in minutes based on word count and language.
    
    Args:
        word_count: Number of words (or characters for CJK)
        language: The language of the text
    
    Returns:
        Estimated reading time in minutes
    """
    if language in ["Chinese", "Japanese", "Korean", "Thai", "Lao", "Khmer", "Telugu", "Malayalam", "Kannada", "Tamil"]:
        # For character-based languages, average reading speed is ~500 chars/min
        return max(1, word_count // 500)
    else:
        # For word-based languages, average reading speed is ~200 words/min
        return max(1, word_count // 200)


def detect_language_from_text(text: str) -> Optional[str]:
    """
    Simple language detection based on character ranges.
    Not 100% accurate but good enough for basic detection.
    
    Args:
        text: The text to analyze
    
    Returns:
        Detected language name or None
    """
    if not text:
        return None
    
    # Remove whitespace and punctuation for analysis
    text_sample = re.sub(r'[\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)[:100]
    
    # Count different character types
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text_sample))
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text_sample))
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text_sample))
    thai_chars = len(re.findall(r'[\u0e00-\u0e7f]', text_sample))
    arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text_sample))
    telugu_chars = len(re.findall(r'[\u0c00-\u0c7f]', text_sample))
    malayalam_chars = len(re.findall(r'[\u0d00-\u0d7f]', text_sample))
    kannada_chars = len(re.findall(r'[\u0c80-\u0cff]', text_sample))
    tamil_chars = len(re.findall(r'[\u0b80-\u0bff]', text_sample))
    
    total_chars = len(text_sample)
    if total_chars == 0:
        return None
    
    # If more than 30% is from a specific script, consider it that language
    threshold = 0.3
    
    if chinese_chars / total_chars > threshold:
        return "Chinese"
    elif japanese_chars / total_chars > threshold:
        return "Japanese"
    elif korean_chars / total_chars > threshold:
        return "Korean"
    elif thai_chars / total_chars > threshold:
        return "Thai"
    elif arabic_chars / total_chars > threshold:
        return "Arabic"
    elif telugu_chars / total_chars > threshold:
        return "Telugu"
    elif malayalam_chars / total_chars > threshold:
        return "Malayalam"
    elif kannada_chars / total_chars > threshold:
        return "Kannada"
    elif tamil_chars / total_chars > threshold:
        return "Tamil"
    
    return "English"  # Default to English


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS by removing HTML tags and normalizing whitespace.
    
    Args:
        text: Raw text with potential HTML
    
    Returns:
        Cleaned text suitable for TTS
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
