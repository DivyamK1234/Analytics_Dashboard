import streamlit as st
import pandas as pd
import os
import re
from language_support import translate_text, LANGUAGES, setup_language_support

def load_article_data():
    """Load article data from CSV file"""
    csv_path = "extracted_content.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error(f"Cannot find article data at {csv_path}")
        return pd.DataFrame()

def display_multilingual_viewer():
    """Main function to display the multilingual content viewer"""
    st.title("Multilingual Content Viewer")
    
    # Setup language support
    current_language = setup_language_support()
    
    # Translate key UI elements
    ui_elements = {
        "title": "Multilingual Content Viewer",
        "select_article": "Select an article to view",
        "original_content": "Original Content",
        "translated_content": "Translated Content",
        "content_language": "Content Language",
        "metadata": "Article Metadata",
        "select_language": "Select language to translate to",
        "no_articles_found": "No articles found. Please make sure extracted_content.csv exists."
    }
    
    # Translate UI elements if not in English
    if current_language != "en":
        for key, text in ui_elements.items():
            ui_elements[key] = translate_text(text, current_language)
    
    # Load article data
    df = load_article_data()
    
    if df.empty:
        st.warning(ui_elements["no_articles_found"])
        return
    
    # Add company column based on source
    df["Company"] = df["Source"].apply(
        lambda x: "KPMG" if isinstance(x, str) and "kpmg" in x.lower() 
        else "PwC" if isinstance(x, str) and "pwc" in x.lower() 
        else "Unknown"
    )
    
    # Display article selector
    article_titles = df["Title"].tolist()
    selected_article_idx = st.selectbox(
        ui_elements["select_article"],
        range(len(article_titles)),
        format_func=lambda idx: f"{article_titles[idx]} ({df['Company'].iloc[idx]})"
    )
    
    # Get selected article
    selected_article = df.iloc[selected_article_idx]
    
    # Display metadata
    st.subheader(ui_elements["metadata"])
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Title:** {selected_article['Title']}")
        st.markdown(f"**Company:** {selected_article['Company']}")
    
    with col2:
        st.markdown(f"**Date Published:** {selected_article['Date Published']}")
        st.markdown(f"**Source:** {selected_article['Source']}")
    
    # Display full content
    st.subheader(ui_elements["content_language"])
    target_language = st.selectbox(
        ui_elements["select_language"],
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index("English") if current_language == "en" else list(LANGUAGES.values()).index(current_language)
    )
    
    # Get the content
    content = selected_article["Content"]
    
    # Split display into tabs for original and translated
    tab1, tab2 = st.tabs([ui_elements["original_content"], ui_elements["translated_content"]])
    
    with tab1:
        st.markdown(content)
    
    with tab2:
        with st.spinner("Translating content..."):
            translated_content = translate_text(content, LANGUAGES[target_language])
            st.markdown(translated_content)
    
    # Display data about translated content
    word_count_original = len(re.findall(r'\w+', content))
    word_count_translated = len(re.findall(r'\w+', translated_content))
    
    st.info(f"Original: {word_count_original} words | Translated: {word_count_translated} words")

if __name__ == "__main__":
    display_multilingual_viewer() 