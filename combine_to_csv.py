import os
import csv
import re

# Directory containing cleaned text files
INPUT_DIR = "cleaned_texts"
# Output CSV file
OUTPUT_FILE = "all_texts.csv"

def main():
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist")
        return
    
    # Get all text files in the input directory, excluding summary files
    text_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt') and not f.startswith('_')]
    print(f"Found {len(text_files)} text files to combine")
    
    # Prepare CSV file
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        # Define CSV columns
        fieldnames = [
            'Filename', 'Title', 'Date Published', 'Source', 
            'Word Count', 'Sentence Count', 'Avg Sentence Length',
            'Top Words', 'Full Content'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each file
        for text_file in text_files:
            file_path = os.path.join(INPUT_DIR, text_file)
            
            try:
                # Read the file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Extract metadata and content
                title = ""
                date = ""
                source = ""
                word_count = ""
                sentence_count = ""
                avg_sentence_length = ""
                top_words = ""
                content = ""
                
                # Parse the file structure
                sections = text.split("--- CLEANED CONTENT ---")
                if len(sections) > 1:
                    metadata_section = sections[0]
                    content = sections[1].strip()
                    
                    # Extract metadata
                    for line in metadata_section.split("\n"):
                        if line.startswith("Title:"):
                            title = line.replace("Title:", "").strip()
                        elif line.startswith("Date Published:"):
                            date = line.replace("Date Published:", "").strip()
                        elif line.startswith("Source:"):
                            source = line.replace("Source:", "").strip()
                        elif line.startswith("Word Count:"):
                            word_count = line.replace("Word Count:", "").strip()
                        elif line.startswith("Sentence Count:"):
                            sentence_count = line.replace("Sentence Count:", "").strip()
                        elif line.startswith("Average Sentence Length:"):
                            avg_sentence_length = line.replace("Average Sentence Length:", "").strip()
                    
                    # Extract top words
                    top_words_section = ""
                    if "--- TOP 20 WORDS ---" in metadata_section:
                        top_words_parts = metadata_section.split("--- TOP 20 WORDS ---")
                        if len(top_words_parts) > 1:
                            top_words_lines = top_words_parts[1].strip().split("\n")
                            top_words = " | ".join([line.strip() for line in top_words_lines if line.strip()])
                
                # Write to CSV
                writer.writerow({
                    'Filename': text_file,
                    'Title': title,
                    'Date Published': date,
                    'Source': source,
                    'Word Count': word_count,
                    'Sentence Count': sentence_count,
                    'Avg Sentence Length': avg_sentence_length,
                    'Top Words': top_words,
                    'Full Content': content
                })
                
                print(f"Processed: {text_file}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                # Write error row
                writer.writerow({
                    'Filename': text_file,
                    'Title': "ERROR",
                    'Date Published': "",
                    'Source': "",
                    'Word Count': "",
                    'Sentence Count': "",
                    'Avg Sentence Length': "",
                    'Top Words': "",
                    'Full Content': f"Error: {str(e)}"
                })
    
    print(f"Successfully combined all files into {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 