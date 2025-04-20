import os
import csv
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import re
import time
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configuration
INPUT_CSV = "all_texts.csv"  # You can also use "texts_metadata.csv" if that's what you have
CLEANED_TEXTS_DIR = "cleaned_texts"
OUTPUT_DIR = "analysis_results"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Make sure to set this in .env file or environment

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure Gemini API
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
generation_model = genai.GenerativeModel('gemini-1.5-flash')

def get_text_from_file(file_path):
    """Read text content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Extract only the content part (after the metadata)
            if "--- CLEANED CONTENT ---" in content:
                content = content.split("--- CLEANED CONTENT ---")[1].strip()
            
            return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def classify_industry(text, max_retries=3):
    """Classify the article into specific industries using Gemini"""
    if not text.strip():
        return {"industry": "Unknown", "confidence": 0.0}
    
    prompt = f"""
    Classify the following article into a specific industry or sector. 
    Consider common sectors like Finance, Technology, Healthcare, Energy, 
    Retail, Manufacturing, etc. Provide your answer in JSON format with 
    'industry' and 'confidence' (a number between 0 and 1) fields.
    If you cannot determine the industry with reasonable confidence, use "Unknown".
    
    Article:
    {text[:5000]}
    
    JSON Response:
    """
    
    for attempt in range(max_retries):
        try:
            response = generation_model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
            
            # Clean up JSON string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            
            try:
                result = json.loads(json_str)
                if isinstance(result, dict) and 'industry' in result:
                    return result
            except json.JSONDecodeError:
                # Try to extract structured data if JSON parsing fails
                industry_match = re.search(r'"industry"\s*:\s*"([^"]+)"', json_str)
                confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', json_str)
                
                if industry_match:
                    industry = industry_match.group(1)
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                    return {"industry": industry, "confidence": confidence}
            
            # If we got here, try a simpler format
            return {"industry": "Unknown", "confidence": 0.0}
        
        except Exception as e:
            print(f"Error in classification (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 * (attempt + 1))  # Exponential backoff
    
    return {"industry": "Unknown", "confidence": 0.0}

def extract_themes_topics(text, max_retries=3):
    """Extract themes, topics, or keywords using Gemini"""
    if not text.strip():
        return {"themes": [], "topics": [], "keywords": []}
    
    prompt = f"""
    Analyze the following article and extract:
    1. Key themes (3-5 high-level concepts)
    2. Specific topics covered (up to 5)
    3. Important keywords (up to 10)
    
    Structure your response in JSON format with 'themes', 'topics', and 'keywords' as lists.
    
    Article:
    {text[:5000]}
    
    JSON Response:
    """
    
    for attempt in range(max_retries):
        try:
            response = generation_model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
            
            # Clean up JSON string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    # Ensure all expected keys exist
                    result['themes'] = result.get('themes', [])
                    result['topics'] = result.get('topics', [])
                    result['keywords'] = result.get('keywords', [])
                    return result
            except json.JSONDecodeError:
                # If JSON parsing fails, return empty results
                pass
            
            # If we got here, create a basic structure
            return {"themes": [], "topics": [], "keywords": []}
        
        except Exception as e:
            print(f"Error extracting themes (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 * (attempt + 1))  # Exponential backoff
    
    return {"themes": [], "topics": [], "keywords": []}

def cluster_by_themes(articles, n_clusters=5):
    """Cluster articles based on their themes and topics rather than embeddings"""
    # Extract themes and topics text for each article
    theme_texts = []
    valid_articles = []
    
    for article in articles:
        themes = article.get('themes', [])
        topics = article.get('topics', [])
        keywords = article.get('keywords', [])
        
        # Combine themes, topics, and keywords into a single text
        theme_text = " ".join(themes + topics + keywords)
        
        # Only include articles with themes/topics
        if theme_text.strip():
            theme_texts.append(theme_text)
            valid_articles.append(article)
    
    if not theme_texts:
        print("No themes found in articles. Cannot cluster.")
        return [], []
    
    # Vectorize the theme texts using TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    theme_vectors = vectorizer.fit_transform(theme_texts)
    
    # Adjust n_clusters if we have fewer samples
    n_clusters = min(n_clusters, len(theme_texts))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(theme_vectors)
    
    return valid_articles, clusters

def visualize_theme_clusters(articles, clusters, output_file):
    """Visualize theme clusters using PCA and save the plot"""
    # Extract themes and topics for each article
    theme_texts = []
    titles = []
    
    for article in articles:
        themes = article.get('themes', [])
        topics = article.get('topics', [])
        keywords = article.get('keywords', [])
        
        theme_text = " ".join(themes + topics + keywords)
        theme_texts.append(theme_text)
        titles.append(article.get('title', 'Untitled')[:30])
    
    # Vectorize the theme texts using TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    theme_vectors = vectorizer.fit_transform(theme_texts)
    
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    reduced_vectors = pca.fit_transform(theme_vectors.toarray())
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'cluster': clusters,
        'title': titles
    })
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get unique clusters and assign colors
    unique_clusters = sorted(df['cluster'].unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot each cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['x'], cluster_data['y'], color=colors[i], 
                   label=f'Cluster {cluster}', alpha=0.7)
    
    # Add labels for points (limit to avoid overcrowding)
    if len(df) <= 20:
        for i, row in df.iterrows():
            plt.annotate(row['title'][:20] + '...', 
                        (row['x'], row['y']),
                        fontsize=8,
                        alpha=0.8)
    
    plt.title('Clusters of Similar Themes (PCA visualization)')
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()

def summarize_theme_clusters(articles, clusters):
    """Generate summaries for each theme cluster"""
    unique_clusters = sorted(set(clusters))
    cluster_summaries = {}
    
    for cluster_id in unique_clusters:
        # Convert NumPy int32 to native Python int 
        cluster_id_key = int(cluster_id) if hasattr(cluster_id, 'item') else cluster_id
        
        # Get articles in this cluster
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_articles = [articles[i] for i in cluster_indices]
        
        # Extract themes, topics and titles for the summary
        all_themes = []
        all_topics = []
        all_keywords = []
        titles = []
        
        for article in cluster_articles:
            all_themes.extend(article.get('themes', []))
            all_topics.extend(article.get('topics', []))
            all_keywords.extend(article.get('keywords', []))
            titles.append(article.get('title', 'Untitled'))
        
        # Count frequency of themes and topics
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Get top themes and topics
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create cluster name from most common theme
        cluster_name = f"Cluster {cluster_id_key}"
        if top_themes:
            cluster_name = f"{top_themes[0][0]} Cluster"
        
        # Store cluster information
        cluster_summaries[cluster_id_key] = {
            "summary": {
                "cluster_name": cluster_name,
                "common_themes": [theme for theme, _ in top_themes],
                "common_topics": [topic for topic, _ in top_topics],
                "common_keywords": [keyword for keyword, _ in top_keywords]
            },
            "articles": [{"title": article.get('title', 'Untitled'), 
                          "url": article.get('url', ''),
                          "industry": article.get('industry', 'Unknown'),
                          "themes": article.get('themes', []),
                          "topics": article.get('topics', [])} 
                         for article in cluster_articles],
            "article_count": len(cluster_articles)
        }
    
    return cluster_summaries

def main():
    print("Starting Comparative Analysis with Gemini API (Theme-based Clustering)...")
    
    # Step 1: Load data - try to load from CSV first
    try:
        # Try to read from the all_texts.csv file
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} articles from {INPUT_CSV}")
        
        # Check if we have the expected columns
        if 'Filename' in df.columns and 'Full Content' not in df.columns:
            # We have filenames but not content, we'll need to load from files
            use_files = True
        else:
            # We have content in the CSV
            use_files = False
    except FileNotFoundError:
        # If CSV not found, try to read from cleaned_texts directory
        if os.path.exists(CLEANED_TEXTS_DIR):
            files = [f for f in os.listdir(CLEANED_TEXTS_DIR) if f.endswith('.txt')]
            print(f"CSV not found. Loaded {len(files)} files from {CLEANED_TEXTS_DIR} directory")
            
            # Create a dataframe from files
            df = pd.DataFrame({
                'Filename': files,
                'Title': [f.replace('.txt', '') for f in files],
                'Full Content': None  # Will be loaded later
            })
            use_files = True
        else:
            raise FileNotFoundError(f"Neither {INPUT_CSV} nor {CLEANED_TEXTS_DIR} directory found.")
    
    # Step 2: Load article content if needed
    articles = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading articles"):
        article = {}
        
        # Get basic metadata from CSV
        for col in df.columns:
            if col in row:
                article[col.lower()] = row[col]
        
        # Load content if needed
        if use_files:
            file_path = os.path.join(CLEANED_TEXTS_DIR, row['Filename'])
            if os.path.exists(file_path):
                article['content'] = get_text_from_file(file_path)
            else:
                print(f"Warning: File not found: {file_path}")
                article['content'] = ""
        else:
            # Content is already in the CSV in the 'Full Content' column
            article['content'] = row.get('Full Content', '')
        
        # Sanitize content
        if isinstance(article['content'], str):
            article['content'] = article['content'].strip()
        else:
            article['content'] = ""
            
        articles.append(article)
    
    print(f"Processed {len(articles)} articles")
    
    # Step 3: Classify articles by industry
    print("Classifying articles by industry...")
    for article in tqdm(articles, desc="Classifying industries"):
        if 'industry' not in article or not article['industry']:
            industry_result = classify_industry(article['content'])
            article['industry'] = industry_result['industry']
            article['industry_confidence'] = industry_result['confidence']
    
    # Step 4: Extract themes, topics, and keywords
    print("Extracting themes, topics, and keywords...")
    for article in tqdm(articles, desc="Extracting themes"):
        themes_result = extract_themes_topics(article['content'])
        article['themes'] = themes_result.get('themes', [])
        article['topics'] = themes_result.get('topics', [])
        article['keywords'] = themes_result.get('keywords', [])
    
    # Step 5: Cluster articles based on themes instead of embeddings
    print("Clustering articles based on themes...")
    valid_articles, clusters = cluster_by_themes(articles, n_clusters=5)
    
    # Add cluster information to articles
    for i, article in enumerate(valid_articles):
        # Convert NumPy int32 to native Python int
        article['cluster'] = int(clusters[i])
    
    # Step 6: Visualize theme clusters
    print("Visualizing theme clusters...")
    visualize_theme_clusters(valid_articles, 
                           clusters,
                           os.path.join(OUTPUT_DIR, "theme_clusters.png"))
    
    # Step 7: Generate cluster summaries
    print("Generating cluster summaries...")
    cluster_summaries = summarize_theme_clusters(valid_articles, clusters)
    
    # Step 8: Save results to JSON files
    # Save article analysis
    with open(os.path.join(OUTPUT_DIR, "article_analysis.json"), 'w', encoding='utf-8') as f:
        # Use a custom JSON encoder to handle NumPy types
        json.dump(valid_articles, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else x)
    
    # Save cluster summaries
    with open(os.path.join(OUTPUT_DIR, "cluster_analysis.json"), 'w', encoding='utf-8') as f:
        json.dump(cluster_summaries, f, indent=2)
    
    # Save industry distribution
    industry_counts = {}
    for article in valid_articles:
        industry = article.get('industry', 'Unknown')
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    
    with open(os.path.join(OUTPUT_DIR, "industry_distribution.json"), 'w', encoding='utf-8') as f:
        json.dump(industry_counts, f, indent=2)
    
    # Save analysis summary to CSV
    with open(os.path.join(OUTPUT_DIR, "analysis_summary.csv"), 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Title', 'URL', 'Industry', 'Cluster', 'Themes', 'Topics', 'Keywords']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for article in valid_articles:
            writer.writerow({
                'Title': article.get('title', 'Untitled'),
                'URL': article.get('url', ''),
                'Industry': article.get('industry', 'Unknown'),
                'Cluster': article.get('cluster', -1),
                'Themes': '; '.join(article.get('themes', [])),
                'Topics': '; '.join(article.get('topics', [])),
                'Keywords': '; '.join(article.get('keywords', []))
            })
    
    print("\nComparative Analysis complete!")
    print(f"Results saved to {OUTPUT_DIR} directory:")
    print(f"  - Article analysis: {os.path.join(OUTPUT_DIR, 'article_analysis.json')}")
    print(f"  - Cluster analysis: {os.path.join(OUTPUT_DIR, 'cluster_analysis.json')}")
    print(f"  - Industry distribution: {os.path.join(OUTPUT_DIR, 'industry_distribution.json')}")
    print(f"  - Analysis summary: {os.path.join(OUTPUT_DIR, 'analysis_summary.csv')}")
    print(f"  - Theme clusters visualization: {os.path.join(OUTPUT_DIR, 'theme_clusters.png')}")

if __name__ == "__main__":
    main() 