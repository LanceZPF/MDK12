import argparse
import json
import sys
import os
import csv
csv.field_size_limit(sys.maxsize)
import path_init
from utils.path_utils import PathUtils
import requests
from tqdm import tqdm

class RephraseQuestion():
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
        # output_dir = os.path.join(parent_dir, "MDK_paraphrase_set")
        

        input_dir = os.path.dirname(self.input_file)
        parent_dir = os.path.dirname(input_dir)

        # 先拿到 input_filename
        input_filename = os.path.basename(self.input_file)

        # 这里先构造 output_dir
        output_dir = os.path.join(parent_dir, f"MDK_paraphrase_set")
        os.makedirs(output_dir, exist_ok=True)

        # 然后再拼装输出文件名
        output_filename = os.path.splitext(input_filename)[0] + "_paraphrase" + os.path.splitext(input_filename)[1]
        self.output_file = os.path.join(output_dir, output_filename)

        
        self.prompt = self.get_prompt("rephrase_cot_k12.txt")
        
        self.examples = self.load_tsv_data(self.input_file)

    def get_prompt(self, prompt_file_name):
        prompt_file = os.path.join(prompt_file_name)
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt
    
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

    def fetch_data_from_openai(self):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.api_key
        }
        
        # Filter examples that need processing
        examples_to_process = [ex for ex in self.examples if "original_question" not in ex]
        
        # Create progress bar
        pbar = tqdm(total=len(examples_to_process), desc="Processing examples")
        
        for i, example in enumerate(examples_to_process):
            prompt_text = f"{self.prompt}\n\nQuestion: {example['question']}\nRephrase the above question: "
            
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
                
                original_question = example['question']
                example['question'] = reply
                example['original_question'] = original_question
            except Exception as e:
                print(e)
                continue
            
            # Update progress bar
            pbar.update(1)
            
            if (i + 1) % 10 == 0:
                self.save_data()
                
        pbar.close()
        self.save_data()
        print("Finished.")


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
        rephrase_cot = RephraseQuestion(args)
        rephrase_cot.fetch_data_from_openai()