"""
Translation Handler

A utility class for handling text translation, language detection, and title cleaning.
Supports automatic translation of non-English text to English.
"""

import re
import demoji
from googletrans import Translator
from langdetect import detect_langs, LangDetectException


class TranslationHandler:
    """
    Handles text translation, language detection, and title cleaning operations.
    
    Supports automatic translation of non-English text to English with
    confidence-based language detection and emoji removal.
    """
    
    # Constants
    LANGUAGE_CONFIDENCE_THRESHOLD = 0.95
    ENGLISH_PATTERN = re.compile(r'^[A-Za-z0-9\s\-\.\,\!\?\:\;\'\"\/\(\)\#\@\&\*\+\=\%\$\[\]\{\}\|\\\~\`\^\<\>\_]+$')
    TITLE_CLEANUP_PATTERN = re.compile(r'[^A-Za-z0-9\s\-\.\,\!\?\:\;\'\"\/\(\)\#\@\&\*\+\=\%\$\[\]\{\}\|\\\~\`\^\<\>\_]+')
    
    def __init__(self):
        """Initialize the translation handler with a Google Translate client."""
        self.translator = Translator()
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of text with confidence threshold.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', etc.)
        """
        try:
            languages = detect_langs(text)
            if not languages:
                return "en"
                
            top_language = languages[0]
            
            # If confidence is low, look for non-English alternatives
            if top_language.prob < self.LANGUAGE_CONFIDENCE_THRESHOLD:
                for lang_obj in languages:
                    if lang_obj.lang != 'en':
                        return lang_obj.lang
                return top_language.lang
            
            # High confidence case - validate English pattern
            is_english = (
                top_language.lang == 'en' and 
                bool(self.ENGLISH_PATTERN.fullmatch(text))
            )
            return "en" if is_english else top_language.lang
            
        except LangDetectException as e:
            print(f"Language detection error: {e}")
            return "en"
    
    async def translate_text(self, text: str, target_language: str = 'en') -> str:
        """
        Translate text to the target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (default: 'en')
            
        Returns:
            Translated text
        """
        try:
            translation = await self.translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    async def clean_title(self, title: str) -> str:
        """
        Clean and normalize video title.
        
        Args:
            title: Raw video title
            
        Returns:
            Cleaned title
        """
        # Remove emojis
        title = demoji.replace(title, "")
        
        # Translate if not English
        if self.detect_language(title) != "en":
            print(f"Translating title: {title}")
            title = await self.translate_text(title, 'en')
        
        # Clean unwanted characters
        return self.TITLE_CLEANUP_PATTERN.sub('', title)
    
    async def aclose(self):
        """Properly close the translation client."""
        client = getattr(self.translator, 'client', None)
        if client is not None and hasattr(client, 'aclose'):
            try:
                await client.aclose()
                print("  ✅ Google Translate client closed")
            except Exception as e:
                print(f"  ⚠️  Warning: Error closing Google Translate client: {e}")
        else:
            print("  ℹ️  No Google Translate client to close")