import argparse
import json
import os
import csv
import path_init
from utils.path_utils import PathUtils
import requests
from tqdm import tqdm
import re
import time

class TypePermutation():
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
        output_dir = os.path.join(parent_dir, "MDK-easy_type_permutation_set")
        os.makedirs(output_dir, exist_ok=True)
        
        input_filename = os.path.basename(self.input_file)
        output_filename = os.path.splitext(input_filename)[0] + "_type_permutation" + os.path.splitext(input_filename)[1]
        self.output_file = os.path.join(output_dir, output_filename)
        
        # Determine the prompt file based on input file name
        if "calculation" in input_filename.lower():
            self.prompt_file = "typepermu_cot_k12_true_false.txt"
            self.conversion_type = "calculation to true/false"
        elif "true_false" in input_filename.lower():
            self.prompt_file = "typepermu_cot_k12_fill_in_the_blank.txt"
            self.conversion_type = "true/false to fill-in-the-blank"
        elif "fill_in_the_blank" in input_filename.lower():
            self.prompt_file = "typepermu_cot_k12_choice.txt"
            self.conversion_type = "fill-in-the-blank to multiple-choice"
        elif "choice" in input_filename.lower():
            self.prompt_file = "typepermu_cot_k12_open-ended.txt"
            self.conversion_type = "multiple-choice to open-ended"
        else:
            print("something wrong")
            exit()
        
        self.prompt = self.get_prompt(self.prompt_file)
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
                        'answer': row[3] if len(row) > 3 else "",  # Get answer if available
                        'row_data': row  # Store the entire row for later reconstruction
                    })
        return examples

    def save_data(self):
        with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            for example in self.examples:
                # Update the question and answer in the original row data
                row_data = example['row_data']
                row_data[1] = example['question']
                if len(row_data) > 3:
                    row_data[3] = example['answer']
                if 'original_question' in example:
                    # Add original question as a new column if needed
                    row_data.append(example['original_question'])
                if 'original_answer' in example:
                    # Add original answer as a new column if needed
                    row_data.append(example['original_answer'])
                tsv_writer.writerow(row_data)

    def convert_question_types(self):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Filter examples that need processing
        examples_to_process = [ex for ex in self.examples if "original_question" not in ex]
        
        # Create progress bar
        pbar = tqdm(total=len(examples_to_process), desc=f"Converting {self.conversion_type}")
        
        for i, example in enumerate(examples_to_process):
            prompt_text = f"{self.prompt}\n\n<Question> {example['question']} </Question>\n<Answer> {example['answer']} </Answer>\n\nConverted question and answer:"
            
            payload = {
                "model": self.eng,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": self.temperature,
                "n": 1
            }
            
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=60
                    )
                    
                    response_data = json.loads(response.text)
                    
                    if 'choices' not in response_data:
                        print(f"API Error: {response_data}")
                        retry_count += 1
                        time.sleep(5)  # Wait before retrying
                        continue
                        
                    reply = response_data['choices'][0]['message']['content'].strip()
                    success = True
                    
                except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
                    print(f"Error occurred: {str(e)}")
                    retry_count += 1
                    time.sleep(5)  # Wait before retrying
            
            if not success:
                print(f"Failed to process example {i+1} after {max_retries} attempts. Skipping.")
                pbar.update(1)
                continue
            
            # Extract new question and answer using regex
            question_match = re.search(r'<Question>(.*?)</Question>', reply, re.DOTALL)
            answer_match = re.search(r'<Answer>(.*?)</Answer>', reply, re.DOTALL)
            
            if question_match and answer_match:
                new_question = question_match.group(1).strip()
                new_answer = answer_match.group(1).strip()
                
                # Store original values
                original_question = example['question']
                original_answer = example['answer']
                
                # Update with new values
                example['question'] = new_question
                example['answer'] = new_answer
                example['original_question'] = original_question
                example['original_answer'] = original_answer
            else:
                print(f"Warning: Could not extract question and answer from response for example {i+1}")
            
            # Update progress bar
            pbar.update(1)
            
            if (i + 1) % 10 == 0:
                self.save_data()
                
        pbar.close()
        self.save_data()
        print(f"Finished converting questions from {self.conversion_type}.")


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
        type_permutation = TypePermutation(args)
        type_permutation.convert_question_types()