import streamlit as st
import requests
import json
import os
from functools import lru_cache

# Define available languages
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi"
}

# Reverse mapping for displaying language names
LANGUAGE_NAMES = {code: name for name, code in LANGUAGES.items()}

# Initialize session state for language if not already set
def init_language_state():
    if 'language' not in st.session_state:
        st.session_state.language = "en"
    if 'language_name' not in st.session_state:
        st.session_state.language_name = "English"

# Function to use Google Translate API
@lru_cache(maxsize=1000)  # Cache translations to avoid repeated API calls
def translate_text(text, target_language="en"):
    """
    Translate text using Google Translate API
    Note: For a production app, you would need to set up a Google Cloud account
    and enable the Translation API
    """
    # For demo purposes, using a simpler approach with HTTP requests
    # In production, you should use the official Google Cloud Translation client library
    try:
        # Free tier approach using a public API (limited usage, for demo only)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_language}&dt=t&q={text}"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the response
            result = response.json()
            translated_text = ''.join([sentence[0] for sentence in result[0]])
            return translated_text
        else:
            print(f"Translation error: {response.status_code}")
            return text  # Return original text if translation fails
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

# Function to get language selector widget
def language_selector():
    init_language_state()
    
    # Create a language dropdown in the sidebar
    selected_language_name = st.sidebar.selectbox(
        "Language / Idioma / Langue",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.language_name)
    )
    
    # Update session state if language changed
    if LANGUAGES[selected_language_name] != st.session_state.language:
        st.session_state.language = LANGUAGES[selected_language_name]
        st.session_state.language_name = selected_language_name
        # Force a rerun to apply the language change
        st.rerun()
    
    return st.session_state.language

# Function to translate UI text
def translate_ui(text_dict):
    """
    Translate a dictionary of UI text elements
    
    Args:
        text_dict: Dictionary with keys as identifiers and values as English text
        
    Returns:
        Dictionary with translated text
    """
    if st.session_state.language == "en":
        return text_dict  # No translation needed for English
    
    translated_dict = {}
    for key, text in text_dict.items():
        translated_dict[key] = translate_text(text, st.session_state.language)
    
    return translated_dict

# Main function to use in your app
def setup_language_support():
    """
    Initialize language support and return the current language code
    """
    init_language_state()
    return language_selector()

# Dictionary of common UI text for easy translation
COMMON_UI_TEXT = {
    # Navigation
    "overview": "Overview",
    "industry_analysis": "Industry Analysis",
    "industry_insights": "Industry Insights",
    "themes_topics": "Themes & Topics",
    "keyword_comparison": "Keyword Comparison",
    "clustering_insights": "Clustering Insights",
    "methodology": "Methodology",
    
    # Section headings
    "dashboard_title": "KPMG vs PWC Comparative Analysis Dashboard",
    "key_insights": "Key Insights",
    "summary_statistics": "Summary Statistics",
    "articles_analyzed": "Articles Analyzed",
    "unique_industries": "Unique Industries",
    "unique_themes": "Unique Themes",
    
    # Common phrases
    "industry": "Industry",
    "topic": "Topic",
    "theme": "Theme",
    "keyword": "Keyword",
    "article_count": "Article Count",
    "compare": "Compare",
} 