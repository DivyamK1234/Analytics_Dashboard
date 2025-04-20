import os
import re
import asyncio
import aiofiles
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string

# Directory containing extracted text files
INPUT_DIR = "extracted_texts"
# Directory for cleaned text files
OUTPUT_DIR = "cleaned_texts"

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Get stopwords
stop_words = set(stopwords.words('english'))

async def clean_text(text):
    """Clean the text content"""
    # Extract the actual content part (after the "--- CONTENT ---" marker)
    content_parts = text.split('--- CONTENT ---\n\n')
    if len(content_parts) > 1:
        content = content_parts[1]
    else:
        content = text
    
    # Remove header/metadata information that might appear in PDFs
    lines = content.split('\n')
    # Skip header lines with page numbers, etc.
    cleaned_lines = []
    for line in lines:
        # Skip empty lines or lines that are likely headers/footers (page numbers, etc.)
        if not line.strip() or re.match(r'^[0-9]+$', line.strip()) or len(line.strip()) < 3:
            continue
        # Skip lines that are likely URLs or email addresses only
        if re.match(r'^(https?://|www\.|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)$', line.strip()):
            continue
        cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # Remove special characters and numbers
    content = re.sub(r'[^\w\s.,;:!?\'"-]', ' ', content)
    
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Normalize spacing around punctuation
    content = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', content)
    
    # Additional cleaning for analysis readiness
    # Remove very short words (likely OCR errors)
    content = re.sub(r'\b\w{1,2}\b', '', content)
    
    # Remove multiple punctuation
    content = re.sub(r'([.,;:!?]){2,}', r'\1', content)
    
    # Fix common OCR errors
    content = re.sub(r'l\b', 'i', content)  # Replace lone 'l' with 'i'
    content = content.replace('0', 'o')  # Replace '0' with 'o'
    
    # Remove lines that are just punctuation or very short
    lines = content.split('\n')
    content = '\n'.join([line for line in lines if len(line.strip()) > 5])
    
    return content

async def clean_and_analyze_file(file_path):
    """Clean and analyze a single text file"""
    try:
        # Read file
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = await f.read()
        
        # Extract metadata from the file
        title = ""
        date = ""
        source = ""
        metadata = {}
        
        lines = text.split('\n')
        for line in lines[:20]:  # Check first 20 lines for metadata
            if line.startswith('Title:'):
                title = line.replace('Title:', '').strip()
                metadata['title'] = title
            elif line.startswith('Date Published:'):
                date = line.replace('Date Published:', '').strip()
                metadata['date'] = date
            elif line.startswith('Source:'):
                source = line.replace('Source:', '').strip()
                metadata['source'] = source
        
        # Clean the text
        cleaned_text = await clean_text(text)
        
        # Generate advanced analytics
        analytics = {}
        
        # Word count
        words = word_tokenize(cleaned_text.lower())
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        analytics['word_count'] = len(filtered_words)
        
        # Sentence count
        sentences = sent_tokenize(cleaned_text)
        analytics['sentence_count'] = len(sentences)
        
        # Average sentence length
        if len(sentences) > 0:
            avg_sentence_length = len(words) / len(sentences)
            analytics['avg_sentence_length'] = round(avg_sentence_length, 2)
        else:
            analytics['avg_sentence_length'] = 0
        
        # Top 20 most frequent words
        word_freq = {}
        for word in filtered_words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:20]
        analytics['top_words'] = top_words
        
        # Generate filename for output
        filename = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Write cleaned text with metadata and analytics
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            # Write metadata
            await f.write(f"Title: {metadata.get('title', '')}\n")
            await f.write(f"Date Published: {metadata.get('date', '')}\n")
            await f.write(f"Source: {metadata.get('source', '')}\n")
            await f.write(f"Word Count: {analytics['word_count']}\n")
            await f.write(f"Sentence Count: {analytics['sentence_count']}\n")
            await f.write(f"Average Sentence Length: {analytics['avg_sentence_length']}\n")
            
            # Write top words
            await f.write("\n--- TOP 20 WORDS ---\n")
            for word, count in analytics['top_words']:
                await f.write(f"{word}: {count}\n")
            
            # Write cleaned content
            await f.write("\n--- CLEANED CONTENT ---\n\n")
            await f.write(cleaned_text)
        
        return {
            'filename': filename,
            'word_count': analytics['word_count'],
            'sentence_count': analytics['sentence_count'],
            'top_words': [word for word, _ in analytics['top_words'][:5]]
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {
            'filename': os.path.basename(file_path),
            'error': str(e)
        }

async def main():
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all text files in the input directory
    text_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    print(f"Found {len(text_files)} text files to process")
    
    # Process each file
    tasks = []
    for text_file in text_files:
        file_path = os.path.join(INPUT_DIR, text_file)
        tasks.append(clean_and_analyze_file(file_path))
    
    results = await asyncio.gather(*tasks)
    
    # Print summary
    print("\nProcessing complete:")
    print(f"Cleaned and analyzed {len(results)} files")
    print(f"Results saved to '{OUTPUT_DIR}' directory")
    
    # Create summary file
    summary_path = os.path.join(OUTPUT_DIR, "_summary.txt")
    async with aiofiles.open(summary_path, 'w', encoding='utf-8') as f:
        await f.write("CONTENT ANALYSIS SUMMARY\n")
        await f.write("=======================\n\n")
        
        for i, result in enumerate(results):
            await f.write(f"Document {i+1}: {result.get('filename')}\n")
            if 'error' in result:
                await f.write(f"  Error: {result['error']}\n")
            else:
                await f.write(f"  Word count: {result.get('word_count')}\n")
                await f.write(f"  Sentence count: {result.get('sentence_count')}\n")
                await f.write(f"  Top 5 words: {', '.join(result.get('top_words', []))}\n")
            await f.write("\n")
    
    print(f"Summary report saved to {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())