import argparse
import json
import sys
import os
import re
import csv
csv.field_size_limit(sys.maxsize)
import path_init
from utils.path_utils import PathUtils
import requests
from tqdm import tqdm

class WordReplacer():
    def __init__(self, args):
        self.args = args
        self.input_file = args.input_file
        self.temperature = args.temp
        self.eng = args.eng
        self.base_url = args.base_url
        self.api_key = args.api_key
        
        # Create output directory and file path
        input_dir = os.path.dirname(self.input_file)
        parent_dir = os.path.dirname(input_dir)
        # os.makedirs(output_dir, exist_ok=True)
        
        input_filename = os.path.basename(self.input_file)

        output_dir = os.path.join(parent_dir, "MDK-easy_substitution_set")
        os.makedirs(output_dir, exist_ok=True)
        filename_without_ext, ext = os.path.splitext(input_filename)
        self.output_file = os.path.join(output_dir, f"{filename_without_ext}_substitution{ext}")
        
        self.examples = self.load_tsv_data(self.input_file)

    def load_tsv_data(self, file_path):
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            for row in tsv_reader:
                if len(row) > 1:  # Ensure row has at least a question field
                    examples.append({
                        'question': row[1],
                        'row_data': row  # Store the entire row for later reconstruction
                    })
        return examples

    def save_data(self):
        with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for example in self.examples:
                # Update the question in the original row data
                row_data = example['row_data']
                row_data[1] = example['question']
                if 'original_question' in example:
                    # Add original question as a new column if needed
                    row_data.append(example['original_question'])
                tsv_writer.writerow(row_data)

    def extract_keywords(self, text):
        """Extract keywords from the given text using API call"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.api_key
        }
        
        prompt_text = f"Extract important keywords from the following text. Return the keywords as a JSON array of strings:\n\n{text}"
        
        payload = {
            "model": self.eng,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature,
            "n": 1
        }

        try:
        
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            
            response_data = json.loads(response.text)
            reply = response_data['choices'][0]['message']['content'].strip()

            try:
                # Extract JSON array from the response
                # Try to find a JSON array in the response
                match = re.search(r'\[.*\]', reply, re.DOTALL)
                if match:
                    keywords = json.loads(match.group(0))
                else:
                    # If no JSON array is found, try to parse the entire response
                    keywords = json.loads(reply)
                
                if not isinstance(keywords, list):
                    keywords = []
            except json.JSONDecodeError:
                # If JSON parsing fails, extract words that appear to be keywords
                keywords = [word.strip('"`\'') for word in re.findall(r'["\'`][^"\'`]+["\'`]', reply)]
                if not keywords:
                    # Fallback to splitting by commas or newlines
                    keywords = [word.strip() for word in re.split(r'[,\n]', reply) if word.strip()]
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return ''
        
        return keywords

    def get_synonyms_batch(self, keywords, max_retries=2):
        """Get synonyms for a list of keywords in a single API call"""
        if not keywords:
            return []
            
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.api_key
        }
        
        keywords_json = json.dumps(keywords)
        prompt_text = f"""For each keyword in the following JSON array, provide a single synonym. 
Return ONLY a JSON array of synonyms in the same order as the input keywords, with no additional text.
Input keywords: {keywords_json}"""
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": 'gpt-4o-mini',
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": self.temperature,
                    "n": 1
                }
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=60
                )
                
                response_data = json.loads(response.text)
                reply = response_data['choices'][0]['message']['content'].strip()
                
                # Try to extract JSON array from the response
                match = re.search(r'\[.*\]', reply, re.DOTALL)
                if match:
                    synonyms = json.loads(match.group(0))
                else:
                    # If no JSON array is found, try to parse the entire response
                    synonyms = json.loads(reply)
                
                # Verify we got the same number of synonyms as keywords
                if len(synonyms) == len(keywords):
                    # Clean up each synonym
                    return [re.sub(r'^["\'`]|["\'`]$|^\s+|\s+$|\.+$', '', syn) for syn in synonyms]
                else:
                    print(f"Attempt {attempt+1}: Got {len(synonyms)} synonyms for {len(keywords)} keywords. Retrying...")
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
        
        print(f"Failed to get synonyms after {max_retries} attempts. Skipping.")
        return []

    def replace_keywords_with_synonyms(self):
        # Filter examples that need processing
        examples_to_process = [ex for ex in self.examples if "original_question" not in ex]
        
        # Create progress bar
        pbar = tqdm(total=len(examples_to_process), desc="Processing examples")
        
        for i, example in enumerate(examples_to_process):
            original_question = example['question']
            modified_question = original_question
            
            # Extract keywords
            keywords = self.extract_keywords(original_question)
            if keywords == '' or not keywords:
                pbar.update(1)
                continue
            
            # Get synonyms for all keywords in a single batch
            synonyms = self.get_synonyms_batch(keywords)
            if not synonyms or len(synonyms) != len(keywords):
                pbar.update(1)
                continue
            
            # Replace each keyword with its synonym
            for keyword, synonym in zip(keywords, synonyms):
                if len(keyword.strip()) > 0 and synonym and synonym != keyword:
                    # Use regex with word boundaries to avoid partial word replacements
                    escaped_keyword = re.escape(keyword)
                    modified_question = re.sub(
                        fr'\b{escaped_keyword}\b', 
                        synonym, 
                        modified_question, 
                        flags=re.IGNORECASE
                    )
            
            example['original_question'] = original_question
            example['question'] = modified_question
            
            # Update progress bar
            pbar.update(1)
            
            if (i + 1) % 10 == 0:
                self.save_data()
                
        pbar.close()
        self.save_data()
        print("Finished replacing keywords with synonyms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', default='/mnt/workspace/zpf/Dynamic-K/MDK12easy', type=str, help="Directory containing TSV files to process")
    parser.add_argument('--eng', default="gpt-4o", type=str)
    parser.add_argument('--temp', default=0.7, type=float)
    parser.add_argument('--base_url', default="XX", type=str, help="OpenAI API base URL")
    parser.add_argument('--api_key', default="XX", type=str, help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Find all TSV files in the specified directory
    tsv_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.tsv'):
                tsv_files.append(os.path.join(root, file))
    
    # Process each TSV file
    for tsv_file in tsv_files:
        print(f"Processing file: {tsv_file}")
        args.input_file = tsv_file
        word_replacer = WordReplacer(args)
        word_replacer.replace_keywords_with_synonyms()