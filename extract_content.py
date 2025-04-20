import os
import csv
import asyncio
import aiohttp
import aiofiles
import PyPDF2
import re
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Directory where PDFs are stored
PDF_DIR = "pdfs"
# CSV file with article URLs
CSV_FILE = "pwc_articles_recent.csv"
# Output file for extracted content
OUTPUT_FILE = "extracted_content.csv"

async def extract_from_pdf(pdf_path):
    """Extract content from a PDF file"""
    try:
        content = ""
        title = ""
        date = ""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract metadata if available
            metadata = pdf_reader.metadata
            if metadata:
                if metadata.title:
                    title = metadata.title
                if metadata.creation_date:
                    date = str(metadata.creation_date)
            
            # If no title in metadata, use filename
            if not title:
                title = os.path.basename(pdf_path).replace('.pdf', '')
            
            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                content += page.extract_text() + "\n"
                
                # Try to find title in first page if not in metadata
                if page_num == 0 and not title:
                    first_page_text = page.extract_text()
                    lines = first_page_text.split('\n')
                    if lines and len(lines) > 0:
                        # First non-empty line is likely the title
                        for line in lines:
                            if line.strip():
                                title = line.strip()
                                break
                
                # Try to find date in first few pages if not in metadata
                if page_num < 3 and not date:
                    page_text = page.extract_text()
                    # Look for date patterns (various formats)
                    date_patterns = [
                        r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b',  # MM/DD/YYYY
                        r'\b\d{1,2}\-\d{1,2}\-\d{2,4}\b',  # MM-DD-YYYY
                        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
                        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
                    ]
                    for pattern in date_patterns:
                        date_match = re.search(pattern, page_text, re.IGNORECASE)
                        if date_match:
                            date = date_match.group(0)
                            break
        
        return {
            'title': title,
            'date': date,
            'content': content,
            'source_file': pdf_path
        }
    except Exception as e:
        print(f"Error extracting content from {pdf_path}: {str(e)}")
        return {
            'title': os.path.basename(pdf_path).replace('.pdf', ''),
            'date': '',
            'content': f"Error extracting content: {str(e)}",
            'source_file': pdf_path
        }

async def extract_from_webpage(session, url):
    """Extract content from a webpage"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title = ""
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text(strip=True)
                else:
                    # Try to find h1 or other heading
                    heading = soup.find(['h1', 'h2'])
                    if heading:
                        title = heading.get_text(strip=True)
                
                # Extract date
                date = ""
                date_element = soup.find(['time', 'span', 'p', 'div'], 
                                        class_=lambda c: c and any(term in (c or '') for term in 
                                                                ['date', 'time', 'publish', 'posted']))
                if date_element:
                    date = date_element.get_text(strip=True)
                    # Also check for datetime attribute
                    if date_element.get('datetime'):
                        date = date_element.get('datetime')
                
                # Extract main content
                content = ""
                # Try to find article or main content div
                main_content = soup.find(['article', 'main', 'div'], 
                                        class_=lambda c: c and any(term in (c or '') for term in 
                                                                ['content', 'article', 'main', 'body']))
                if main_content:
                    # Remove scripts, styles, and nav elements
                    for element in main_content.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        element.decompose()
                    content = main_content.get_text(separator='\n', strip=True)
                else:
                    # Just get the body text if can't find specific content area
                    body = soup.find('body')
                    if body:
                        for element in body.find_all(['script', 'style', 'nav', 'header', 'footer']):
                            element.decompose()
                        content = body.get_text(separator='\n', strip=True)
                    else:
                        content = soup.get_text(separator='\n', strip=True)
                
                return {
                    'title': title,
                    'date': date,
                    'content': content,
                    'source_url': url
                }
            else:
                return {
                    'title': f"Failed to access: {url}",
                    'date': '',
                    'content': f"Error: HTTP status {response.status}",
                    'source_url': url
                }
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return {
            'title': f"Error processing: {url}",
            'date': '',
            'content': f"Error: {str(e)}",
            'source_url': url
        }

async def main():
    # Create a list to store all content
    all_content = []
    
    # Create a session for HTTP requests
    async with aiohttp.ClientSession() as session:
        # Process PDFs
        pdf_tasks = []
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            pdf_tasks.append(extract_from_pdf(pdf_path))
        
        # Process article URLs from CSV
        article_tasks = []
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    url = row.get('Article URL', '')
                    # Skip PDF URLs as we're processing them directly
                    if url and not url.lower().endswith('.pdf'):
                        article_tasks.append(extract_from_webpage(session, url))
        
        # Run all tasks
        pdf_results = await asyncio.gather(*pdf_tasks)
        article_results = await asyncio.gather(*article_tasks)
        
        # Combine results
        all_content.extend(pdf_results)
        all_content.extend(article_results)
    
    # Write all content to CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Date Published', 'Source', 'Content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in all_content:
            source = item.get('source_file', item.get('source_url', ''))
            writer.writerow({
                'Title': item.get('title', ''),
                'Date Published': item.get('date', ''),
                'Source': source,
                'Content': item.get('content', '')[:1000] + '...' if len(item.get('content', '')) > 1000 else item.get('content', '')
            })
    
    print(f"Processed PDFs and articles")
    print(f"Extracted content saved to {OUTPUT_FILE}")
    
    # Also save full content in separate text files
    os.makedirs("extracted_texts", exist_ok=True)
    for i, item in enumerate(all_content):
        # Determine the filename based on the source
        source = item.get('source_file', item.get('source_url', ''))
        
        if 'source_file' in item and item['source_file']:
            # For PDFs, use the original PDF filename but with .txt extension
            pdf_filename = os.path.basename(item['source_file'])
            filename = f"extracted_texts/{pdf_filename.replace('.pdf', '.txt')}"
        else:
            # For web articles, use the title to create a filename
            title = item.get('title', f"item_{i}")
            # Clean up title for filename
            safe_title = re.sub(r'[^\w\s-]', '', title)
            safe_title = re.sub(r'[\s-]+', '_', safe_title)
            safe_title = safe_title[:50]  # Limit length
            filename = f"extracted_texts/{safe_title}.txt"
        
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(f"Title: {item.get('title', '')}\n")
            await f.write(f"Date Published: {item.get('date', '')}\n")
            await f.write(f"Source: {source}\n")
            await f.write("\n--- CONTENT ---\n\n")
            await f.write(item.get('content', ''))
    
    print(f"Full text content saved to individual files in the 'extracted_texts' directory")

if __name__ == "__main__":
    asyncio.run(main()) 