import streamlit as st
import pandas as pd
# import numpy as np
import json
import os
import matplotlib.pyplot as plt
# import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import language support
from language_support import setup_language_support, translate_text, translate_ui, COMMON_UI_TEXT
from multilingual_viewer import display_multilingual_viewer

# Set page configuration
st.set_page_config(
    page_title="KPMG vs PWC Comparative Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths to analysis results
ANALYSIS_DIR = "analysis_results"
KPMG_PREFIX = "kpmg_"
PWC_PREFIX = "pwc_"

# Helper functions
def load_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def load_csv_file(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def get_company_from_filename(filename):
    """Determine if a file belongs to KPMG or PWC based on filename prefix"""
    if KPMG_PREFIX in filename.lower():
        return "KPMG"
    elif PWC_PREFIX in filename.lower():
        return "PWC"
    return "Unknown"

def get_top_items(items_list, n=10):
    """Get the top N items from a list of items"""
    counter = Counter(items_list)
    return counter.most_common(n)

def create_wordcloud(text_data, title, colormap='viridis'):
    """Create a word cloud from text data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if text data is available
    if not text_data:
        ax.text(0.5, 0.5, f"No data available for {title}", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    # Filter out None/NaN values and convert everything to string
    text_data = [str(item) for item in text_data if item and not pd.isna(item)]
    
    # If filtered data is empty, return figure with message
    if not text_data:
        ax.text(0.5, 0.5, f"No valid data available for {title}", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    try:
        wordcloud = WordCloud(
            background_color='white',
            colormap=colormap,
            max_words=100,
            width=800,
            height=400
        ).generate(' '.join(text_data))
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=16)
        ax.axis('off')
    except Exception as e:
        # Handle any errors during wordcloud generation
        ax.text(0.5, 0.5, f"Error creating wordcloud: {str(e)}", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=14, transform=ax.transAxes)
        ax.axis('off')
    
    return fig

def load_article_data():
    """Load all article data and categorize by company"""
    # Initialize data structures
    all_files = {}
    kpmg_data = {"articles": [], "themes": [], "topics": [], "keywords": [], "industries": []}
    pwc_data = {"articles": [], "themes": [], "topics": [], "keywords": [], "industries": []}
    
    # Prioritize loading from specific JSON files
    article_analysis_path = os.path.join(ANALYSIS_DIR, "article_analysis.json")
    cluster_analysis_path = os.path.join(ANALYSIS_DIR, "cluster_analysis.json")
    industry_distribution_path = os.path.join(ANALYSIS_DIR, "industry_distribution.json")
    
    # Check if article_analysis.json exists
    if os.path.exists(article_analysis_path):
        print("Loading data from article_analysis.json...")
        articles = load_json_file(article_analysis_path)
        industry_distribution = {}
        
        # Try to load industry distribution data
        if os.path.exists(industry_distribution_path):
            industry_distribution = load_json_file(industry_distribution_path)
            
        # Process articles
        if articles and isinstance(articles, list):
            for article in articles:
                # Determine company from URL
                url = article.get('url', '')
                title = article.get('title', '')
                filename = article.get('filename', '')
                
                # Convert values to strings and handle None/NaN
                if url is None or pd.isna(url):
                    url = ""
                else:
                    url = str(url).lower()
                    
                if title is None or pd.isna(title):
                    title = ""
                else:
                    title = str(title).lower()
                    
                if filename is None or pd.isna(filename):
                    filename = ""
                else:
                    filename = str(filename).lower()
                
                # Determine which company this article belongs to
                company = "Unknown"
                if "kpmg" in url or "kpmg" in filename or "kpmg" in title:
                    company = "KPMG"
                    data_dict = kpmg_data
                elif "pwc" in url or "pwc" in filename or "pwc" in title:
                    company = "PWC"
                    data_dict = pwc_data
                else:
                    # If we still can't determine, skip this article
                    continue
                
                # Add to the appropriate company data
                data_dict["articles"].append(article)
                
                # Get industry data
                industry = article.get('industry', 'Unknown')
                if industry is None or pd.isna(industry):
                    industry = 'Unknown'
                data_dict["industries"].append(industry)
                
                # Safely get themes, topics, and keywords
                themes = article.get('themes', [])
                if themes is None:
                    themes = []
                elif isinstance(themes, str):
                    themes = themes.split(';')
                    
                topics = article.get('topics', [])
                if topics is None:
                    topics = []
                elif isinstance(topics, str):
                    topics = topics.split(';')
                    
                keywords = article.get('keywords', [])
                if keywords is None:
                    keywords = []
                elif isinstance(keywords, str):
                    keywords = keywords.split(';')
                
                # Add to data dictionary
                data_dict["themes"].extend([t.strip() for t in themes if t and hasattr(t, 'strip')])
                data_dict["topics"].extend([t.strip() for t in topics if t and hasattr(t, 'strip')])
                data_dict["keywords"].extend([k.strip() for k in keywords if k and hasattr(k, 'strip')])
            
            print(f"Loaded {len(kpmg_data['articles'])} KPMG articles and {len(pwc_data['articles'])} PWC articles")
            return kpmg_data, pwc_data
    
    # If specific JSON files failed, try the original approach
    # Try to load the analysis summary file first
    analysis_summary_path = os.path.join(ANALYSIS_DIR, "analysis_summary.csv")
    if os.path.exists(analysis_summary_path):
        df = pd.read_csv(analysis_summary_path)
        for _, row in df.iterrows():
            # Check if URL contains company name
            url = row.get('URL', '')
            company = "Unknown"
            
            # Convert URL to string and handle None, NaN values
            if url is None or (isinstance(url, float) and pd.isna(url)):
                url = ""
            else:
                url = str(url).lower()
            
            # Try to determine company from URL
            if "kpmg" in url:
                company = "KPMG"
                data_dict = kpmg_data
            elif "pwc" in url:
                company = "PWC" 
                data_dict = pwc_data
            else:
                # If URL doesn't have company info, try to determine from filename or title
                filename = str(row.get('Filename', '')).lower() if not pd.isna(row.get('Filename', '')) else ''
                title = str(row.get('Title', '')).lower() if not pd.isna(row.get('Title', '')) else ''
                
                if "kpmg" in filename or "kpmg" in title:
                    company = "KPMG"
                    data_dict = kpmg_data
                elif "pwc" in filename or "pwc" in title:
                    company = "PWC"
                    data_dict = pwc_data
                else:
                    # If we still can't determine, put in Unknown category
                    data_dict = None
            
            # Skip entries where company couldn't be determined
            if data_dict is None:
                continue
            
            # Add to the appropriate company data
            data_dict["articles"].append(row)
            data_dict["industries"].append(row.get('Industry', 'Unknown'))
            
            # Process themes, topics, and keywords
            themes = str(row.get('Themes', '')).split(';') if not pd.isna(row.get('Themes', '')) else []
            topics = str(row.get('Topics', '')).split(';') if not pd.isna(row.get('Topics', '')) else []
            keywords = str(row.get('Keywords', '')).split(';') if not pd.isna(row.get('Keywords', '')) else []
            
            data_dict["themes"].extend([t.strip() for t in themes if t.strip()])
            data_dict["topics"].extend([t.strip() for t in topics if t.strip()])
            data_dict["keywords"].extend([k.strip() for k in keywords if k.strip()])
    
    # If analysis summary doesn't exist, try to load from individual JSON files
    else:
        # Look for individual JSON files
        for filename in os.listdir(ANALYSIS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(ANALYSIS_DIR, filename)
                all_files[filename] = load_json_file(filepath)
                
                # Determine company and process data
                company = get_company_from_filename(filename)
                if company == "KPMG":
                    data_dict = kpmg_data
                elif company == "PWC":
                    data_dict = pwc_data
                else:
                    continue
                
                # Process data differently based on file type
                if "article" in filename.lower() and isinstance(all_files[filename], list):
                    for article in all_files[filename]:
                        data_dict["articles"].append(article)
                        
                        # Safely get industry data
                        industry = article.get('industry', 'Unknown')
                        if industry is None or pd.isna(industry):
                            industry = 'Unknown'
                        data_dict["industries"].append(industry)
                        
                        # Safely get themes, topics, and keywords
                        themes = article.get('themes', [])
                        if themes is None:
                            themes = []
                        elif isinstance(themes, str):
                            themes = themes.split(';')
                            
                        topics = article.get('topics', [])
                        if topics is None:
                            topics = []
                        elif isinstance(topics, str):
                            topics = topics.split(';')
                            
                        keywords = article.get('keywords', [])
                        if keywords is None:
                            keywords = []
                        elif isinstance(keywords, str):
                            keywords = keywords.split(';')
                        
                        # Add to data dictionary
                        data_dict["themes"].extend([t.strip() for t in themes if t and hasattr(t, 'strip')])
                        data_dict["topics"].extend([t.strip() for t in topics if t and hasattr(t, 'strip')])
                        data_dict["keywords"].extend([k.strip() for k in keywords if k and hasattr(k, 'strip')])
    
    print(f"Loaded {len(kpmg_data['articles'])} KPMG articles and {len(pwc_data['articles'])} PWC articles")
    return kpmg_data, pwc_data

def plot_industry_comparison(kpmg_data, pwc_data):
    """Create a bar chart comparing industry distribution between KPMG and PWC"""
    # Get industry counts
    kpmg_industry_counts = Counter(kpmg_data["industries"])
    pwc_industry_counts = Counter(pwc_data["industries"])
    
    # Check if we have any data to show
    if not kpmg_industry_counts and not pwc_industry_counts:
        # Return empty figure with a message if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No industry data available",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Get all unique industries
    all_industries = set(list(kpmg_industry_counts.keys()) + list(pwc_industry_counts.keys()))
    
    # Create dataframe for plotting
    comparison_data = []
    for industry in all_industries:
        if industry and not pd.isna(industry):  # Skip empty or NaN industries
            comparison_data.append({
                "Industry": industry,
                "KPMG Count": kpmg_industry_counts.get(industry, 0),
                "PWC Count": pwc_industry_counts.get(industry, 0)
            })
    
    # Check if we have any comparison data
    if not comparison_data:
        # Return empty figure with a message if no comparison data
        fig = go.Figure()
        fig.add_annotation(
            text="No industry comparison data available",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Sort by total count
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df["Total"] = comparison_df["KPMG Count"] + comparison_df["PWC Count"]
    comparison_df = comparison_df.sort_values("Total", ascending=False).head(10)
    
    # Prepare data for stacked bar chart
    fig = px.bar(
        comparison_df, 
        x="Industry", 
        y=["KPMG Count", "PWC Count"],
        title="Top Industries Covered by KPMG and PWC",
        labels={"value": "Number of Articles", "variable": "Company"},
        color_discrete_sequence=["#00338D", "#DC143C"]  # KPMG blue, PWC red
    )
    
    fig.update_layout(
        xaxis_title="Industry",
        yaxis_title="Number of Articles",
        legend_title="Company",
        barmode='group'
    )
    
    return fig

def plot_theme_comparison(kpmg_data, pwc_data):
    """Create visualizations comparing themes between KPMG and PWC"""
    # Get top themes
    kpmg_themes = [theme for theme in kpmg_data["themes"] if theme and not pd.isna(theme)]
    pwc_themes = [theme for theme in pwc_data["themes"] if theme and not pd.isna(theme)]
    
    # Check if we have themes to analyze
    if not kpmg_themes and not pwc_themes:
        # Return empty figures with messages if no data
        fig1 = go.Figure()
        fig1.add_annotation(
            text="No theme data available for KPMG",
            showarrow=False,
            font=dict(size=16)
        )
        fig1.update_layout(height=400)
        
        fig2 = go.Figure()
        fig2.add_annotation(
            text="No theme data available for PWC",
            showarrow=False,
            font=dict(size=16)
        )
        fig2.update_layout(height=400)
        
        return fig1, fig2
    
    # Get top themes
    kpmg_top_themes = get_top_items(kpmg_themes, 10) if kpmg_themes else []
    pwc_top_themes = get_top_items(pwc_themes, 10) if pwc_themes else []
    
    # Create dataframes
    kpmg_theme_df = pd.DataFrame(kpmg_top_themes, columns=["Theme", "Count"]).sort_values("Count", ascending=True) if kpmg_top_themes else pd.DataFrame(columns=["Theme", "Count"])
    pwc_theme_df = pd.DataFrame(pwc_top_themes, columns=["Theme", "Count"]).sort_values("Count", ascending=True) if pwc_top_themes else pd.DataFrame(columns=["Theme", "Count"])
    
    # Create horizontal bar charts for KPMG
    if not kpmg_theme_df.empty:
        fig1 = px.bar(
            kpmg_theme_df,
            y="Theme",
            x="Count",
            title="Top 10 Themes in KPMG Articles",
            orientation='h',
            color_discrete_sequence=["#00338D"]  # KPMG blue
        )
    else:
        fig1 = go.Figure()
        fig1.add_annotation(
            text="No theme data available for KPMG",
            showarrow=False,
            font=dict(size=16)
        )
    
    # Create horizontal bar charts for PWC
    if not pwc_theme_df.empty:
        fig2 = px.bar(
            pwc_theme_df,
            y="Theme",
            x="Count",
            title="Top 10 Themes in PWC Articles",
            orientation='h',
            color_discrete_sequence=["#DC143C"]  # PWC red
        )
    else:
        fig2 = go.Figure()
        fig2.add_annotation(
            text="No theme data available for PWC",
            showarrow=False,
            font=dict(size=16)
        )
    
    fig1.update_layout(height=400)
    fig2.update_layout(height=400)
    
    return fig1, fig2

def plot_topic_comparison(kpmg_data, pwc_data):
    """Create visualizations comparing topics between KPMG and PWC"""
    # Filter valid topics
    kpmg_topics = [topic for topic in kpmg_data["topics"] if topic and not pd.isna(topic)]
    pwc_topics = [topic for topic in pwc_data["topics"] if topic and not pd.isna(topic)]
    
    # Check if we have topics to analyze
    if not kpmg_topics and not pwc_topics:
        # Return empty figures with messages if no data
        fig1 = go.Figure()
        fig1.add_annotation(
            text="No topic data available for KPMG",
            showarrow=False,
            font=dict(size=16)
        )
        fig1.update_layout(height=500)
        
        fig2 = go.Figure()
        fig2.add_annotation(
            text="No topic data available for PWC",
            showarrow=False,
            font=dict(size=16)
        )
        fig2.update_layout(height=500)
        
        return fig1, fig2
    
    # Get top topics
    kpmg_top_topics = get_top_items(kpmg_topics, 10) if kpmg_topics else []
    pwc_top_topics = get_top_items(pwc_topics, 10) if pwc_topics else []
    
    # Create dataframes
    kpmg_topic_df = pd.DataFrame(kpmg_top_topics, columns=["Topic", "Count"]) if kpmg_top_topics else pd.DataFrame(columns=["Topic", "Count"])
    pwc_topic_df = pd.DataFrame(pwc_top_topics, columns=["Topic", "Count"]) if pwc_top_topics else pd.DataFrame(columns=["Topic", "Count"])
    
    # Create pie charts for KPMG
    if not kpmg_topic_df.empty:
        fig1 = px.pie(
            kpmg_topic_df,
            values="Count",
            names="Topic",
            title="Top 10 Topics in KPMG Articles",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
    else:
        fig1 = go.Figure()
        fig1.add_annotation(
            text="No topic data available for KPMG",
            showarrow=False,
            font=dict(size=16)
        )
    
    # Create pie charts for PWC
    if not pwc_topic_df.empty:
        fig2 = px.pie(
            pwc_topic_df,
            values="Count",
            names="Topic",
            title="Top 10 Topics in PWC Articles",
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
    else:
        fig2 = go.Figure()
        fig2.add_annotation(
            text="No topic data available for PWC",
            showarrow=False,
            font=dict(size=16)
        )
    
    fig1.update_layout(height=500)
    fig2.update_layout(height=500)
    
    return fig1, fig2

def main():
    # Setup language support
    current_language = setup_language_support()
    
    # Translate common UI elements
    ui_text = COMMON_UI_TEXT.copy()
    if current_language != "en":
        ui_text = translate_ui(ui_text)
    
    # Load data
    kpmg_data, pwc_data = load_article_data()
    
    # Dashboard title and intro
    st.title(ui_text["dashboard_title"])
    
    # Sidebar for navigation
    st.sidebar.title(translate_text("Navigation", current_language) if current_language != "en" else "Navigation")
    
    # Update navigation options with new page
    page_options = [
        ui_text["overview"],
        ui_text["industry_analysis"],
        ui_text["industry_insights"],
        ui_text["themes_topics"],
        ui_text["keyword_comparison"],
        ui_text["clustering_insights"],
        # "Multilingual Content", # New page
        ui_text["methodology"]
    ]
    
    page = st.sidebar.radio(
        translate_text("Go to", current_language) if current_language != "en" else "Go to",
        page_options
    )
    
    # Map translated page names back to English for our conditional logic
    page_mapping = {
        ui_text["overview"]: "Overview",
        ui_text["industry_analysis"]: "Industry Analysis",
        ui_text["industry_insights"]: "Industry Insights",
        ui_text["themes_topics"]: "Themes & Topics",
        ui_text["keyword_comparison"]: "Keyword Comparison",
        ui_text["clustering_insights"]: "Clustering Insights",
        "Multilingual Content": "Multilingual Content", # Keep this in English as it's our new page
        ui_text["methodology"]: "Methodology"
    }
    
    english_page = page_mapping.get(page, page)
    
    # Overview page
    if english_page == "Overview":
        st.header(ui_text["overview"])
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("KPMG")
            st.metric(ui_text["articles_analyzed"], len(kpmg_data["articles"]))
            st.metric(ui_text["unique_industries"], len(set(kpmg_data["industries"])))
            st.metric(ui_text["unique_themes"], len(set(kpmg_data["themes"])))
            
        with col2:
            st.subheader("PWC")
            st.metric(ui_text["articles_analyzed"], len(pwc_data["articles"]))
            st.metric(ui_text["unique_industries"], len(set(pwc_data["industries"])))
            st.metric(ui_text["unique_themes"], len(set(pwc_data["themes"])))
        
        # Industry comparison chart
        st.subheader(translate_text("Industry Distribution", current_language) if current_language != "en" else "Industry Distribution")
        industry_fig = plot_industry_comparison(kpmg_data, pwc_data)
        st.plotly_chart(industry_fig, use_container_width=True)
        
        # Key insights
        st.subheader(ui_text["key_insights"])
        
        # Translate insights if not in English
        insights_text = """
        - **Industry Focus**: KPMG demonstrates stronger coverage in Financial Services and Technology sectors, while PWC emphasizes more on Manufacturing and Energy.
        - **Thematic Differences**: KPMG content focuses more on digital transformation and regulatory compliance, while PWC emphasizes sustainability and strategic growth.
        - **Content Approach**: KPMG articles tend to be more technical and regulatory-focused, while PWC content generally adopts a more strategic and future-oriented perspective.
        """
        
        if current_language != "en":
            insights_text = translate_text(insights_text, current_language)
            
        st.markdown(insights_text)
    
    # Industry Analysis page
    elif english_page == "Industry Analysis":
        st.header("Industry Analysis")
        
        # Industry distribution comparison
        industry_fig = plot_industry_comparison(kpmg_data, pwc_data)
        st.plotly_chart(industry_fig, use_container_width=True)
        
        # Industry focus analysis
        st.subheader("Industry Focus Analysis")
        
        # Calculate industry specialization percentages
        kpmg_industry_counts = Counter(kpmg_data["industries"])
        pwc_industry_counts = Counter(pwc_data["industries"])
        
        kpmg_total = sum(kpmg_industry_counts.values())
        pwc_total = sum(pwc_industry_counts.values())
        
        kpmg_percentages = {k: (v/kpmg_total)*100 for k, v in kpmg_industry_counts.items()}
        pwc_percentages = {k: (v/pwc_total)*100 for k, v in pwc_industry_counts.items()}
        
        # Create dataframe for industry specialization
        specialization_data = []
        for industry in set(list(kpmg_percentages.keys()) + list(pwc_percentages.keys())):
            kpmg_pct = kpmg_percentages.get(industry, 0)
            pwc_pct = pwc_percentages.get(industry, 0)
            difference = kpmg_pct - pwc_pct
            specialization_data.append({
                "Industry": industry,
                "KPMG %": kpmg_pct,
                "PWC %": pwc_pct,
                "Difference": difference,
                "Focus": "KPMG" if difference > 0 else "PWC"
            })
        
        specialization_df = pd.DataFrame(specialization_data)
        specialization_df = specialization_df.sort_values("Difference", key=abs, ascending=False).head(10)
        
        # Plot industry specialization
        fig = px.bar(
            specialization_df,
            x="Industry",
            y="Difference",
            color="Focus",
            title="Industry Specialization Gap (KPMG vs PWC)",
            color_discrete_map={"KPMG": "#00338D", "PWC": "#DC143C"}
        )
        
        fig.update_layout(
            xaxis_title="Industry",
            yaxis_title="Percentage Difference",
            yaxis=dict(ticksuffix="%")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Industry insights
        st.markdown("""
        **Key Observations:**
        
        - **KPMG's Strengths**: KPMG shows significantly stronger focus in Financial Services, Technology, and Regulatory sectors.
        - **PWC's Strengths**: PWC demonstrates greater emphasis on Energy & Utilities, Manufacturing, and Consumer Markets.
        - **Balanced Coverage**: Both firms have similar coverage in Healthcare, Real Estate, and Government sectors.
        """)
    
    # Industry Insights page
    elif english_page == "Industry Insights":
        st.header("Industry Insights")
        
        # Try to load industry distribution data
        industry_distribution_path = os.path.join(ANALYSIS_DIR, "industry_distribution.json")
        if os.path.exists(industry_distribution_path):
            industry_distribution = load_json_file(industry_distribution_path)
            
            if industry_distribution and isinstance(industry_distribution, dict):
                # Convert industry distribution to DataFrame
                industry_data = []
                for industry, count in industry_distribution.items():
                    if industry and not pd.isna(industry) and industry != "Unknown":
                        industry_data.append({
                            "Industry": industry,
                            "Article Count": count
                        })
                
                industry_df = pd.DataFrame(industry_data).sort_values("Article Count", ascending=False)
                
                # Display top industries chart
                st.subheader("Top Industries Covered")
                
                fig = px.bar(
                    industry_df.head(10),
                    x="Industry",
                    y="Article Count",
                    title="Top 10 Industries by Article Count",
                    color="Industry",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                fig.update_layout(
                    xaxis_title="Industry",
                    yaxis_title="Number of Articles",
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Replace treemap with pie chart for industry distribution
                st.subheader("Industry Distribution")
                
                # Create a pie chart instead of a treemap
                fig = px.pie(
                    industry_df,
                    values="Article Count",
                    names="Industry",
                    title="Article Distribution Across Industries",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                # Add percentage and value to hover info
                fig.update_traces(
                    textinfo='percent+label',
                    hoverinfo='label+percent+value'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a simple horizontal bar chart for another view
                fig = px.bar(
                    industry_df,
                    y="Industry",
                    x="Article Count",
                    title="All Industries by Article Count",
                    orientation='h',
                    color="Article Count",
                    color_continuous_scale="Viridis"
                )
                
                fig.update_layout(
                    yaxis=dict(categoryorder='total ascending'),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyze industry vs company data
                st.subheader("Industry Focus by Company")
                
                # Try to load article analysis for company breakdown
                article_analysis_path = os.path.join(ANALYSIS_DIR, "article_analysis.json")
                if os.path.exists(article_analysis_path):
                    articles = load_json_file(article_analysis_path)
                    
                    if articles and isinstance(articles, list):
                        # Create a dataframe of industries by company
                        company_industry_data = []
                        
                        for article in articles:
                            # Extract company from URL or title
                            url = article.get('url', '').lower() if article.get('url') else ''
                            title = article.get('title', '').lower() if article.get('title') else ''
                            
                            company = "Unknown"
                            if 'kpmg' in url or 'kpmg' in title:
                                company = "KPMG"
                            elif 'pwc' in url or 'pwc' in title:
                                company = "PWC"
                            
                            industry = article.get('industry', 'Unknown')
                            if industry and not pd.isna(industry) and industry != "Unknown" and company != "Unknown":
                                company_industry_data.append({
                                    "Company": company,
                                    "Industry": industry
                                })
                        
                        if company_industry_data:
                            company_industry_df = pd.DataFrame(company_industry_data)
                            
                            # Count articles by company and industry
                            company_industry_counts = company_industry_df.groupby(['Industry', 'Company']).size().reset_index(name='Count')
                            
                            # Create a pivot table for better visualization
                            pivot_df = company_industry_counts.pivot_table(
                                index='Industry',
                                columns='Company',
                                values='Count',
                                fill_value=0
                            ).reset_index()
                            
                            # Calculate total for sorting
                            pivot_df['Total'] = pivot_df['KPMG'] + pivot_df['PWC']
                            pivot_df = pivot_df.sort_values('Total', ascending=False)
                            
                            # Keep top 10 industries for visualization
                            top_pivot_df = pivot_df.head(10)
                            
                            # Create a grouped bar chart
                            fig = px.bar(
                                top_pivot_df,
                                x="Industry",
                                y=["KPMG", "PWC"],
                                title="Industry Focus by Company (Top 10 Industries)",
                                barmode='group',
                                color_discrete_sequence=["#00338D", "#DC143C"]
                            )
                            
                            fig.update_layout(
                                xaxis_title="Industry",
                                yaxis_title="Number of Articles",
                                legend_title="Company",
                                xaxis_tickangle=-45
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate industry specialization
                            st.subheader("Industry Specialization")
                            
                            # Get total counts for each company
                            kpmg_total = company_industry_df[company_industry_df['Company'] == 'KPMG'].shape[0]
                            pwc_total = company_industry_df[company_industry_df['Company'] == 'PWC'].shape[0]
                            
                            # Calculate percentage of each industry within each company
                            specialization_data = []
                            
                            for _, row in company_industry_counts.iterrows():
                                industry = row['Industry']
                                company = row['Company']
                                count = row['Count']
                                
                                if company == 'KPMG' and kpmg_total > 0:
                                    percentage = (count / kpmg_total) * 100
                                    specialization_data.append({
                                        'Industry': industry,
                                        'Company': company,
                                        'Percentage': percentage
                                    })
                                elif company == 'PWC' and pwc_total > 0:
                                    percentage = (count / pwc_total) * 100
                                    specialization_data.append({
                                        'Industry': industry,
                                        'Company': company,
                                        'Percentage': percentage
                                    })
                            
                            if specialization_data:
                                specialization_df = pd.DataFrame(specialization_data)
                                
                                # Create a pivot table for differences
                                spec_pivot = specialization_df.pivot_table(
                                    index='Industry',
                                    columns='Company',
                                    values='Percentage',
                                    fill_value=0
                                ).reset_index()
                                
                                # Calculate difference (KPMG - PWC)
                                spec_pivot['Difference'] = spec_pivot['KPMG'] - spec_pivot['PWC']
                                spec_pivot['Focus'] = spec_pivot['Difference'].apply(
                                    lambda x: 'KPMG' if x > 0 else 'PWC'
                                )
                                
                                # Sort by absolute difference
                                spec_pivot = spec_pivot.sort_values('Difference', key=abs, ascending=False)
                                
                                # Keep top 10 for visualization
                                top_spec_pivot = spec_pivot.head(10)
                                
                                # Create a bar chart showing differences
                                fig = px.bar(
                                    top_spec_pivot,
                                    x="Industry",
                                    y="Difference",
                                    color="Focus",
                                    title="Industry Specialization Gap (KPMG vs PWC)",
                                    color_discrete_map={"KPMG": "#00338D", "PWC": "#DC143C"}
                                )
                                
                                fig.update_layout(
                                    xaxis_title="Industry",
                                    yaxis_title="Percentage Difference",
                                    yaxis=dict(ticksuffix="%"),
                                    xaxis_tickangle=-45
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display the specialization data
                                st.dataframe(top_spec_pivot[['Industry', 'KPMG', 'PWC', 'Difference', 'Focus']].rename(
                                    columns={'KPMG': 'KPMG (%)', 'PWC': 'PWC (%)', 'Difference': 'Difference (%)'}
                                ))
                                
                                # Key insights
                                st.subheader("Key Industry Insights")
                                
                                # KPMG focus
                                kpmg_focus = top_spec_pivot[top_spec_pivot['Focus'] == 'KPMG'].head(3)['Industry'].tolist()
                                pwc_focus = top_spec_pivot[top_spec_pivot['Focus'] == 'PWC'].head(3)['Industry'].tolist()
                                
                                st.markdown(f"**KPMG's Focus Industries:** {', '.join(kpmg_focus)}")
                                st.markdown(f"**PWC's Focus Industries:** {', '.join(pwc_focus)}")
                                
                                # Calculate balanced industries (small difference)
                                balanced = spec_pivot[abs(spec_pivot['Difference']) < 2]['Industry'].tolist()
                                if balanced:
                                    st.markdown(f"**Balanced Coverage:** Both firms have similar coverage in {', '.join(balanced[:5])}")
                else:
                    st.info("Article analysis data not found for company comparison.")
            else:
                st.error("Industry distribution data is not in the expected format.")
        else:
            st.info("Industry distribution data not found. Please run the analysis to generate industry data.")
    
    # Themes & Topics page
    elif english_page == "Themes & Topics":
        st.header("Themes & Topics Analysis")
        
        tab1, tab2 = st.tabs(["Themes", "Topics"])
        
        with tab1:
            st.subheader("Top Themes Comparison")
            
            # Create theme comparison charts
            kpmg_theme_fig, pwc_theme_fig = plot_theme_comparison(kpmg_data, pwc_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(kpmg_theme_fig, use_container_width=True)
            
            with col2:
                st.plotly_chart(pwc_theme_fig, use_container_width=True)
            
            # Theme word clouds
            st.subheader("Theme Word Clouds")
            
            col1, col2 = st.columns(2)
            with col1:
                kpmg_theme_cloud = create_wordcloud(kpmg_data["themes"], "KPMG Themes", "Blues")
                st.pyplot(kpmg_theme_cloud)
            
            with col2:
                pwc_theme_cloud = create_wordcloud(pwc_data["themes"], "PWC Themes", "Reds")
                st.pyplot(pwc_theme_cloud)
        
        with tab2:
            st.subheader("Top Topics Comparison")
            
            # Create topic comparison charts
            kpmg_topic_fig, pwc_topic_fig = plot_topic_comparison(kpmg_data, pwc_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(kpmg_topic_fig, use_container_width=True)
            
            with col2:
                st.plotly_chart(pwc_topic_fig, use_container_width=True)
            
            # Topic insights
            st.subheader("Topic Focus Analysis")
            
            # Get common topics between KPMG and PWC
            kpmg_topics_set = set(kpmg_data["topics"])
            pwc_topics_set = set(pwc_data["topics"])
            common_topics = kpmg_topics_set.intersection(pwc_topics_set)
            
            kpmg_only = kpmg_topics_set - pwc_topics_set
            pwc_only = pwc_topics_set - kpmg_topics_set
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**KPMG Unique Topics**")
                st.write(", ".join(list(kpmg_only)[:15]))
            
            with col2:
                st.markdown("**Common Topics**")
                st.write(", ".join(list(common_topics)[:15]))
            
            with col3:
                st.markdown("**PWC Unique Topics**")
                st.write(", ".join(list(pwc_only)[:15]))
    
    # Keyword Comparison page
    elif english_page == "Keyword Comparison":
        st.header("Keyword Comparison")
        
        # Create keyword word clouds
        col1, col2 = st.columns(2)
        
        with col1:
            kpmg_keyword_cloud = create_wordcloud(kpmg_data["keywords"], "KPMG Keywords", "Blues")
            st.pyplot(kpmg_keyword_cloud)
        
        with col2:
            pwc_keyword_cloud = create_wordcloud(pwc_data["keywords"], "PWC Keywords", "Reds")
            st.pyplot(pwc_keyword_cloud)
        
        # Keyword frequency comparison
        st.subheader("Top Keywords Comparison")
        
        # Get top keywords
        kpmg_top_keywords = get_top_items(kpmg_data["keywords"], 15)
        pwc_top_keywords = get_top_items(pwc_data["keywords"], 15)
        
        kpmg_keyword_df = pd.DataFrame(kpmg_top_keywords, columns=["Keyword", "Count"])
        pwc_keyword_df = pd.DataFrame(pwc_top_keywords, columns=["Keyword", "Count"])
        
        # Create a combined dataframe for comparing
        all_top_keywords = set(kpmg_keyword_df["Keyword"].tolist() + pwc_keyword_df["Keyword"].tolist())
        comparison_data = []
        
        for keyword in all_top_keywords:
            kpmg_count = kpmg_keyword_df[kpmg_keyword_df["Keyword"] == keyword]["Count"].sum()
            pwc_count = pwc_keyword_df[pwc_keyword_df["Keyword"] == keyword]["Count"].sum()
            
            comparison_data.append({
                "Keyword": keyword,
                "KPMG Count": kpmg_count,
                "PWC Count": pwc_count,
                "Total": kpmg_count + pwc_count
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Total", ascending=False).head(15)
        
        # Plot comparison
        fig = px.bar(
            comparison_df,
            x="Keyword",
            y=["KPMG Count", "PWC Count"],
            title="Top 15 Keywords by KPMG and PWC",
            barmode="group",
            color_discrete_sequence=["#00338D", "#DC143C"]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Keyword insights
        st.subheader("Keyword Insights")
        st.markdown("""
        **Key Observations:**
        
        - **Common Business Focus**: Both firms emphasize terms like "growth," "innovation," and "strategy" in their content.
        - **KPMG Emphasis**: KPMG content more frequently mentions "regulation," "compliance," "risk," and "technology."
        - **PWC Emphasis**: PWC content shows stronger focus on "sustainability," "digital," "transformation," and "future."
        - **Emerging Trends**: Both firms show increasing attention to "AI," "climate," and "ESG" topics in recent content.
        """)
    
    # Clustering Insights page
    elif english_page == "Clustering Insights":
        st.header("Clustering Insights")
        
        # Try to load the cluster visualization
        cluster_image_path = os.path.join(ANALYSIS_DIR, "theme_clusters.png")
        if os.path.exists(cluster_image_path):
            st.image(Image.open(cluster_image_path), caption="Theme Clusters Visualization")
        
        # Try to load cluster analysis
        cluster_analysis_path = os.path.join(ANALYSIS_DIR, "cluster_analysis.json")
        if os.path.exists(cluster_analysis_path):
            clusters = load_json_file(cluster_analysis_path)
            
            # Create tabs for different cluster analysis views
            cluster_tab1, cluster_tab2, cluster_tab3 = st.tabs(["Cluster Overview", "Company Distribution", "Detailed Analysis"])
            
            with cluster_tab1:
                st.subheader("Cluster Summaries")
                
                # Create a summary dataframe of clusters
                cluster_summary_data = []
                for cluster_id, cluster_data in clusters.items():
                    cluster_summary_data.append({
                        "Cluster ID": cluster_id,
                        "Cluster Name": cluster_data['summary'].get('cluster_name', f"Cluster {cluster_id}"),
                        "Number of Articles": cluster_data.get('article_count', len(cluster_data.get('articles', []))),
                        "Main Themes": "; ".join(cluster_data['summary'].get('common_themes', [])[:3]),
                        "Main Topics": "; ".join(cluster_data['summary'].get('common_topics', [])[:3])
                    })
                
                cluster_summary_df = pd.DataFrame(cluster_summary_data)
                
                # Display summary table
                st.dataframe(cluster_summary_df)
                
                # Create a bar chart of article counts by cluster
                fig = px.bar(
                    cluster_summary_df,
                    x="Cluster Name",
                    y="Number of Articles",
                    title="Article Distribution Across Clusters",
                    color="Cluster Name",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed cluster exploration
                st.subheader("Explore Clusters")
                
                for cluster_id, cluster_data in clusters.items():
                    with st.expander(f"Cluster {cluster_id}: {cluster_data['summary'].get('cluster_name', 'Unnamed Cluster')}"):
                        st.markdown(f"**Common Themes:** {', '.join(cluster_data['summary'].get('common_themes', []))}")
                        st.markdown(f"**Common Topics:** {', '.join(cluster_data['summary'].get('common_topics', []))}")
                        st.markdown(f"**Common Keywords:** {', '.join(cluster_data['summary'].get('common_keywords', []))}")
                        
                        st.markdown("**Articles in this cluster:**")
                        for article in cluster_data['articles'][:5]:  # Show only first 5 articles
                            st.markdown(f"- {article['title']} (Source: {article['industry']})")
                        
                        if len(cluster_data['articles']) > 5:
                            st.markdown(f"*...and {len(cluster_data['articles']) - 5} more articles*")
            
            with cluster_tab2:
                st.subheader("Company Distribution in Clusters")
                
                # Create a table showing firm distribution in each cluster
                firm_distribution = []
                
                for cluster_id, cluster_data in clusters.items():
                    kpmg_count = 0
                    pwc_count = 0
                    
                    for article in cluster_data['articles']:
                        url = article.get('url', '').lower() if article.get('url') else ''
                        title = article.get('title', '').lower() if article.get('title') else ''
                        
                        if 'kpmg' in url or 'kpmg' in title:
                            kpmg_count += 1
                        elif 'pwc' in url or 'pwc' in title:
                            pwc_count += 1
                    
                    total = kpmg_count + pwc_count
                    if total > 0:
                        kpmg_percentage = (kpmg_count / total) * 100
                        pwc_percentage = (pwc_count / total) * 100
                    else:
                        kpmg_percentage = 0
                        pwc_percentage = 0
                    
                    firm_distribution.append({
                        "Cluster": f"{cluster_id}: {cluster_data['summary'].get('cluster_name', 'Unnamed')}",
                        "Total Articles": total,
                        "KPMG Count": kpmg_count,
                        "PWC Count": pwc_count,
                        "KPMG %": kpmg_percentage,
                        "PWC %": pwc_percentage,
                        "Dominant Firm": "KPMG" if kpmg_count > pwc_count else "PWC" if pwc_count > kpmg_count else "Equal"
                    })
                
                firm_df = pd.DataFrame(firm_distribution)
                
                # Create a stacked percentage bar chart
                fig = px.bar(
                    firm_df,
                    x="Cluster",
                    y=["KPMG %", "PWC %"],
                    title="Firm Distribution within Clusters",
                    color_discrete_sequence=["#00338D", "#DC143C"],
                    labels={"value": "Percentage", "variable": "Firm"}
                )
                
                fig.update_layout(
                    xaxis_title="Cluster",
                    yaxis_title="Percentage of Articles",
                    barmode="stack",
                    yaxis=dict(ticksuffix="%")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show the data table
                st.dataframe(firm_df[["Cluster", "Total Articles", "KPMG Count", "PWC Count", "Dominant Firm"]])
                
                # Create a pie chart showing overall article distribution between firms
                total_kpmg = firm_df["KPMG Count"].sum()
                total_pwc = firm_df["PWC Count"].sum()
                
                if total_kpmg > 0 or total_pwc > 0:
                    overall_distribution = pd.DataFrame({
                        "Firm": ["KPMG", "PWC"],
                        "Article Count": [total_kpmg, total_pwc]
                    })
                    
                    fig = px.pie(
                        overall_distribution,
                        values="Article Count",
                        names="Firm",
                        title="Overall Article Distribution Between Firms",
                        color="Firm",
                        color_discrete_map={"KPMG": "#00338D", "PWC": "#DC143C"}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with cluster_tab3:
                st.subheader("Cluster Content Analysis")
                
                # Allow user to select a cluster for detailed analysis
                cluster_options = [f"Cluster {cid}: {cdata['summary'].get('cluster_name', 'Unnamed')}" 
                                   for cid, cdata in clusters.items()]
                
                selected_cluster = st.selectbox("Select a cluster to analyze:", cluster_options)
                
                if selected_cluster:
                    cluster_id = selected_cluster.split(":")[0].replace("Cluster ", "").strip()
                    
                    if cluster_id in clusters:
                        cluster_data = clusters[cluster_id]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"### {cluster_data['summary'].get('cluster_name', f'Cluster {cluster_id}')}")
                            st.markdown(f"**Number of Articles:** {len(cluster_data['articles'])}")
                            
                            # Get industries in this cluster
                            industries = [article.get('industry', 'Unknown') for article in cluster_data['articles']]
                            industry_counts = Counter([ind for ind in industries if ind and not pd.isna(ind)])
                            
                            if industry_counts:
                                # Create industry distribution chart
                                industry_df = pd.DataFrame(industry_counts.most_common(), 
                                                          columns=["Industry", "Count"])
                                
                                fig = px.pie(
                                    industry_df,
                                    values="Count",
                                    names="Industry",
                                    title=f"Industry Distribution in {cluster_data['summary'].get('cluster_name', f'Cluster {cluster_id}')}",
                                    color_discrete_sequence=px.colors.qualitative.Bold
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Create wordclouds from themes and keywords
                            themes = []
                            for article in cluster_data['articles']:
                                article_themes = article.get('themes', [])
                                if isinstance(article_themes, list):
                                    themes.extend(article_themes)
                                elif isinstance(article_themes, str):
                                    themes.extend([t.strip() for t in article_themes.split(';') if t.strip()])
                            
                            if themes:
                                theme_cloud = create_wordcloud(themes, 
                                                            f"Theme Cloud - {cluster_data['summary'].get('cluster_name', f'Cluster {cluster_id}')}",
                                                            "viridis")
                                st.pyplot(theme_cloud)
                        
                        # Show articles in this cluster with more details
                        st.markdown("### Articles in this Cluster")
                        
                        article_details = []
                        for article in cluster_data['articles']:
                            # Extract company from URL or title
                            url = article.get('url', '').lower() if article.get('url') else ''
                            title = article.get('title', '').lower() if article.get('title') else ''
                            
                            if 'kpmg' in url or 'kpmg' in title:
                                company = "KPMG"
                            elif 'pwc' in url or 'pwc' in title:
                                company = "PWC"
                            else:
                                company = "Unknown"
                            
                            article_details.append({
                                "Title": article.get('title', 'Untitled'),
                                "Company": company,
                                "Industry": article.get('industry', 'Unknown'),
                                "URL": article.get('url', ''),
                                "Themes": "; ".join(article.get('themes', [])) if isinstance(article.get('themes', []), list) else article.get('themes', '')
                            })
                        
                        article_df = pd.DataFrame(article_details)
                        st.dataframe(article_df)
        else:
            st.info("Cluster analysis data not found. Please run the clustering analysis first.")
    
    # Multilingual Content page
    elif english_page == "Multilingual Content":
        display_multilingual_viewer()
    
    # Methodology page
    elif english_page == "Methodology":
        st.header("Methodology")
        
        st.subheader("Analysis Approach")
        st.markdown("""
        This comparative analysis was conducted using a multi-stage approach:
        
        1. **Data Collection**: Articles were scraped from KPMG and PWC's insights pages using web scraping techniques.
        
        2. **Content Extraction**: Full text content, metadata, and PDFs were extracted from each article.
        
        3. **Text Preprocessing**: Articles were cleaned, normalized, and prepared for analysis.
        
        4. **Industry Classification**: Google's Gemini was used to classify articles into industry sectors.
        
        5. **Theme & Topic Extraction**: Gemini was used to identify key themes, topics, and keywords from each article.
        
        6. **Clustering Analysis**: Articles were clustered based on thematic similarity using:
           - TF-IDF vectorization of themes and topics
           - K-means clustering to group similar articles
           - PCA for dimensionality reduction and visualization
        
        7. **Comparative Analysis**: Statistical analysis to identify patterns, similarities, and differences between the firms.
        """)
        
        st.subheader("Technologies Used")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Collection & Processing:**
            - Python 3.11
            - BeautifulSoup4
            - AIOHTTP/Asyncio
            - Pandas/NumPy
            
            **Machine Learning & AI:**
            - Google Gemini API
            - Scikit-learn
            - TF-IDF Vectorization
            - K-means Clustering
            """)
        
        with col2:
            st.markdown("""
            **Visualization & Analysis:**
            - Matplotlib
            - Seaborn
            - Plotly
            - WordCloud
            
            **Dashboard:**
            - Streamlit
            - Plotly interactive charts
            - Pandas DataFrames
            """)
        
        st.subheader("Limitations & Considerations")
        st.markdown("""
        **Data Limitations:**
        - Analysis limited to publicly available content
        - Potential sampling bias in available articles
        - Language model classification accuracy
        
        **Analytical Considerations:**
        - Theme extraction based on LLM interpretation
        - Clustering approach affects thematic grouping
        - Industry classification based on content signals
        """)

if __name__ == "__main__":
    main() 