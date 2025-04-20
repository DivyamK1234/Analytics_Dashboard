import asyncio
import csv
import os
import aiohttp
import aiofiles
import re
from datetime import datetime, timedelta
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Get the current date and calculate dates for filtering
current_date = datetime.now()
two_months_ago = current_date - timedelta(days=60)
print(f"Filtering for dates after: {two_months_ago.strftime('%d/%m/%Y')}")

async def download_pdf(session, url, filename, folder="pdfs"):
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Full path for saving the file
    filepath = os.path.join(folder, filename)
    
    try:
        async with session.get(url) as response:
            if response.status == 200:
                # Check if it's a PDF
                content_type = response.headers.get('Content-Type', '')
                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    # Save the PDF
                    async with aiofiles.open(filepath, 'wb') as f:
                        await f.write(await response.read())
                    print(f"Downloaded: {filename}")
                    return True
                else:
                    print(f"Not a PDF: {url}")
                    # Check if it's an HTML page that might have PDF links
                    if 'text/html' in content_type:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        pdf_links = []
                        for a in soup.find_all('a', href=True):
                            href = a['href']
                            if href.lower().endswith('.pdf'):
                                pdf_url = urljoin(url, href)
                                pdf_filename = pdf_url.split('/')[-1]
                                await download_pdf(session, pdf_url, pdf_filename, folder)
                    return False
            else:
                print(f"Failed to download: {url} (Status: {response.status})")
                return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def is_from_2025(date_str):
    """Check if a date string contains 2025"""
    if not date_str:
        return False
    
    # Check for common date formats
    if '/25' in date_str or '/2025' in date_str or '2025' in date_str:
        return True
    
    # Try to parse and check date more formally
    try:
        # Try different date formats
        for fmt in ['%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d-%m-%y', '%d %b %Y', '%d %B %Y']:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                if parsed_date.year == 2025:
                    return True
            except ValueError:
                continue
    except Exception:
        # If any parsing error, fall back to simple check
        pass
    
    return False

def is_within_last_two_months(date_str):
    """Check if a date string is within the last two months"""
    if not date_str:
        return False
    
    # Try to parse the date from various formats
    try:
        # Try different date formats
        parsed_date = None
        for fmt in ['%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d-%m-%y', '%d %b %Y', '%d %B %Y']:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        
        if parsed_date:
            return parsed_date >= two_months_ago
            
    except Exception:
        # If any parsing error, try simple checks
        pass
    
    # If can't parse, check if it's a very recent date by checking the month
    current_month = current_date.month
    prev_month = (current_date.month - 1) or 12  # If January (1), previous is December (12)
    
    # Check for month names or numbers in date string
    months = {
        1: ['jan', 'january', '01/', '/01/', '-01-'],
        2: ['feb', 'february', '02/', '/02/', '-02-'],
        3: ['mar', 'march', '03/', '/03/', '-03-'],
        4: ['apr', 'april', '04/', '/04/', '-04-'],
        5: ['may', '05/', '/05/', '-05-'],
        6: ['jun', 'june', '06/', '/06/', '-06-'],
        7: ['jul', 'july', '07/', '/07/', '-07-'],
        8: ['aug', 'august', '08/', '/08/', '-08-'],
        9: ['sep', 'september', '09/', '/09/', '-09-'],
        10: ['oct', 'october', '10/', '/10/', '-10-'],
        11: ['nov', 'november', '11/', '/11/', '-11-'],
        12: ['dec', 'december', '12/', '/12/', '-12-']
    }
    
    # Check for current month or previous month indicators
    date_str_lower = date_str.lower()
    if any(indicator in date_str_lower for indicator in months[current_month]):
        return True
    if any(indicator in date_str_lower for indicator in months[prev_month]):
        return True
    
    return False

async def extract_article_info(session, url, only_2025=True, only_recent=True):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Try to find date information
                date_element = soup.find(['time', 'span', 'p', 'div'], 
                                        class_=lambda c: c and any(term in (c or '') for term in 
                                                                ['date', 'time', 'publish', 'posted']))
                
                date_str = date_element.get_text(strip=True) if date_element else ""
                # Also look for datetime attribute
                if date_element and date_element.get('datetime'):
                    date_str = date_element.get('datetime')
                
                # If only_2025 and date was found but not from 2025, skip this article
                if only_2025 and date_str and not is_from_2025(date_str):
                    print(f"Skipping article not from 2025: {date_str}: {url}")
                    return {'url': url, 'pdf_links': [], 'date': date_str, 'is_2025': False, 'is_recent': False}
                
                # If only_recent and date was found but not recent, skip this article
                is_recent = is_within_last_two_months(date_str)
                if only_recent and date_str and not is_recent:
                    print(f"Skipping article not from last 2 months: {date_str}: {url}")
                    return {'url': url, 'pdf_links': [], 'date': date_str, 'is_2025': is_from_2025(date_str), 'is_recent': False}
                
                # Look for PDF download links
                pdf_links = []
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    # Check if it's a PDF link
                    if href.lower().endswith('.pdf'):
                        pdf_url = urljoin(url, href)
                        pdf_links.append(pdf_url)
                    # Sometimes PDFs are linked through buttons or other elements
                    elif any(term in href.lower() for term in ['download', 'view', 'pdf', 'report', 'publication']):
                        pdf_url = urljoin(url, href)
                        pdf_links.append(pdf_url)
                
                # Return article info with PDF links
                return {
                    'url': url,
                    'pdf_links': pdf_links,
                    'date': date_str,
                    'is_2025': is_from_2025(date_str),
                    'is_recent': is_recent
                }
            else:
                print(f"Failed to access article page: {url}")
                return {'url': url, 'pdf_links': [], 'date': "", 'is_2025': False, 'is_recent': False}
    except Exception as e:
        print(f"Error processing article {url}: {str(e)}")
        return {'url': url, 'pdf_links': [], 'date': "", 'is_2025': False, 'is_recent': False}

async def main():
    # Create directories
    os.makedirs("pdfs", exist_ok=True)
    
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on the PWC insights URL
        result = await crawler.arun(url="https://www.pwc.in/research-insights.html")
        
        # Extract article URLs from the page HTML
        article_info = []
        if hasattr(result, 'html'):
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Process collection items (like the example provided)
            collection_items = soup.find_all(['article', 'div'], class_=lambda c: c and any(term in (c or '') for term in 
                                                                               ['collection__item', 'feedItem', 'card', 'article']))
            
            for item in collection_items:
                # Try to find the date
                date_element = item.find(['time', 'span', 'p'], class_=lambda c: c and any(term in (c or '') for term in 
                                                                         ['date', 'time', 'publish']))
                
                date_str = ""
                if date_element:
                    date_str = date_element.get_text(strip=True)
                    # Also check for datetime attribute
                    if date_element.get('datetime'):
                        date_str = date_element.get('datetime')
                
                # Check if from 2025 and recent
                if date_str:
                    if not is_from_2025(date_str):
                        print(f"Skipping non-2025 article: {date_str}")
                        continue
                    if not is_within_last_two_months(date_str):
                        print(f"Skipping article not from last 2 months: {date_str}")
                        continue
                
                # Find the link
                link_tag = item.find('a', href=True)
                if link_tag:
                    href = link_tag['href']
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin("https://www.pwc.in", href)
                    
                    # Try to extract title
                    title_tag = item.find(['h2', 'h3', 'h4', 'h5', 'span', 'div'], class_=lambda c: c and any(term in (c or '') for term in 
                                                                            ['heading', 'title', 'collection__item-heading']))
                    title = title_tag.get_text(strip=True) if title_tag else href.split('/')[-1]
                    
                    # Add to article info with date
                    article_info.append({
                        'url': href,
                        'title': title,
                        'date': date_str
                    })
            
            # Find specific link elements like levelTwoLink
            navigation_links = soup.find_all('a', class_=lambda c: c and any(cls in (c or '') for cls in ['levelTwoLink', 'nav', 'navigation']))
            for link in navigation_links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin("https://www.pwc.in", href)
                    
                    # Check if URL contains 2025 and looks recent
                    if '2025' in href:
                        title = link.get_text(strip=True)
                        article_info.append({
                            'url': href,
                            'title': title,
                            'date': '2025'  # Assume it's from 2025 if in URL
                        })
            
            # Find additional article cards
            article_cards = soup.find_all('div', class_=lambda c: c and ('card' in c or 'article' in c))
            
            # Process the article cards to extract information
            for card in article_cards:
                # Try to find the date
                date_element = card.find(['time', 'span', 'p'], class_=lambda c: c and any(term in (c or '') for term in 
                                                                         ['date', 'time', 'publish']))
                
                date_str = ""
                if date_element:
                    date_str = date_element.get_text(strip=True)
                    # Also check for datetime attribute
                    if date_element.get('datetime'):
                        date_str = date_element.get('datetime')
                
                # Skip if not from 2025 or not recent
                if date_str:
                    if not is_from_2025(date_str):
                        continue
                    if not is_within_last_two_months(date_str):
                        continue
                
                # Try to find the link to the article
                link_tag = card.find('a', href=True)
                if link_tag:
                    href = link_tag['href']
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin("https://www.pwc.in", href)
                    
                    # Try to extract title
                    title_tag = card.find(['h2', 'h3', 'h4', 'h5', 'div', 'span'], class_=lambda c: c and 'title' in (c or ''))
                    title = title_tag.get_text(strip=True) if title_tag else href.split('/')[-1]
                    
                    # Add to article info
                    article_info.append({
                        'url': href,
                        'title': title,
                        'date': date_str
                    })
            
            # If still not enough articles found, look for links with certain patterns
            if len(article_info) < 5:
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    # Skip if already in our list
                    if any(info['url'] == href or info['url'] == urljoin("https://www.pwc.in", href) for info in article_info):
                        continue
                    
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin("https://www.pwc.in", href)
                    
                    # Check if the link might be from recent 2025
                    if '2025' in href:
                        title = link.get_text(strip=True) or href.split('/')[-1]
                        article_info.append({
                            'url': href,
                            'title': title,
                            'date': '2025'  # Assume it's 2025 based on URL
                        })
        
        # Deduplicate articles by URL
        unique_articles = []
        seen_urls = set()
        for article in article_info:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        article_info = unique_articles
        print(f"Found {len(article_info)} potential articles")
        
        # Process each article to find PDFs and verify dates
        async with aiohttp.ClientSession() as session:
            # Visit each article page to find PDF links and verify dates
            tasks = [extract_article_info(session, article['url'], only_2025=True, only_recent=True) for article in article_info]
            results = await asyncio.gather(*tasks)
            
            # Filter for recent 2025 articles
            recent_results = [result for result in results if (result['is_2025'] or '2025' in result['url']) 
                              and (result['is_recent'] or '2025' in result['url'])]
            print(f"Filtered to {len(recent_results)} articles from the last 2 months of 2025")
            
            # Combine results
            pdf_links = []
            for result in recent_results:
                for pdf_url in result['pdf_links']:
                    pdf_filename = pdf_url.split('/')[-1]
                    pdf_links.append({
                        'article_url': result['url'],
                        'pdf_url': pdf_url,
                        'pdf_filename': pdf_filename
                    })
            
            print(f"Found {len(pdf_links)} PDF links from recent articles")
            
            # Also check direct PDF URLs that might not be linked in article pages
            direct_pdf_urls = []
            for article in article_info:
                url = article['url']
                if url.lower().endswith('.pdf') and ('2025' in url or is_from_2025(article.get('date', ''))):
                    pdf_filename = url.split('/')[-1]
                    direct_pdf_urls.append({
                        'article_url': url,
                        'pdf_url': url,
                        'pdf_filename': pdf_filename
                    })
            
            pdf_links.extend(direct_pdf_urls)
            print(f"Added {len(direct_pdf_urls)} direct PDF URLs")
            
            # Save recent article URLs to CSV
            with open('pwc_articles_recent.csv', 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Article URL', 'Title', 'Date', 'PDF URLs'])  # Header
                for result in recent_results:
                    pdf_urls = ';'.join(result['pdf_links'])
                    article_title = next((a['title'] for a in article_info if a['url'] == result['url']), "")
                    writer.writerow([result['url'], article_title, result['date'], pdf_urls])
            
            print(f"Saved {len(recent_results)} recent article URLs to pwc_articles_recent.csv")
            
            # Download PDFs
            if pdf_links:
                download_tasks = []
                for pdf in pdf_links:
                    download_tasks.append(download_pdf(session, pdf['pdf_url'], pdf['pdf_filename']))
                
                # For links that might lead to pages with PDFs but aren't PDFs themselves
                for result in recent_results:
                    url = result['url']
                    if not url.lower().endswith('.pdf') and ('budget' in url.lower() or '2025' in url):
                        filename = f"landing_{url.split('/')[-1]}.html"
                        download_tasks.append(download_pdf(session, url, filename))
                
                download_results = await asyncio.gather(*download_tasks)
                successful_downloads = sum(1 for r in download_results if r)
            else:
                print("No PDF links from recent articles found to download")

# Run the async main function
asyncio.run(main())