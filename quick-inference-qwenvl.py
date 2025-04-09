import argparse
import json
import os
import random
import time
import re
import pandas as pd
import glob

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."

def evaluate_chat_model():
    random.seed(args.seed)
    
    # Find all TSV files in the data_root directory
    tsv_files = glob.glob(os.path.join(args.data_root, "*.tsv"))
    
    if not tsv_files:
        print(f"No TSV files found in {args.data_root}")
        return
    
    for tsv_file in tsv_files:
        ds_name = os.path.basename(tsv_file).split('.')[0]
        print(f"Processing dataset: {ds_name}")
        
        # Read TSV file
        data = []
        df = pd.read_csv(tsv_file, sep='\t')
        for _, row in df.iterrows():
            data.append(row.to_dict())
        
        inputs = []
        for data_item in data:
            question = data_item["question"]
            data_item['query'] = question

            if 'image' not in data_item or pd.isna(data_item['image']):
                image_data = None
                # Handle case when there's no image data
                if image_data is None:
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": SYSTEM_PROMPT
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": data_item['query']
                                },
                            ],
                        }
                    ]
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": None,
                    })
                
            else:
                image_data = "data:image base64," + data_item['image']  # This is already base64 encoded
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_data  # Pass base64 image directly
                            },
                            {
                                "type": "text",
                                "text": data_item['query']
                            },
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Process the base64 image data directly
                image_data_processed, _ = process_vision_info(messages)
                
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image_data_processed
                    },
                })
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids, skip_special_tokens=False)
        start_time = time.time()
        print(f"Starting generation for {len(inputs)} examples...")
        
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        
        end_time = time.time()
        generation_time = (end_time - start_time) / 60.0
        print(f"Generation completed in {generation_time:.2f} minutes")
        
        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            data_item['response'] = model_output.outputs[0].text
            outputs.append(data_item)

        # Create DataFrame for Excel output
        df_data = []
        for data_item in outputs:
            row = {
                'index': data_item.get('index', ''),
                'question': data_item.get('question', ''),
                'category': data_item.get('category', ''),
                'answer': data_item.get('answer', ''),
                'knowledge': data_item.get('knowledge', ''),
                'analysis': data_item.get('analysis', ''),
                'year': data_item.get('year', ''),
                'grade_level': data_item.get('grade_level', ''),
                'difficulty_level': data_item.get('difficulty_level', ''),
                'prediction': data_item.get('response', '')
            }
            df_data.append(row)
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(df_data)
        
        # Generate output filename
        model_name = os.path.basename(args.checkpoint)
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        excel_file = f'Eureka_{ds_name}.xlsx'
        excel_path = os.path.join(args.out_dir, excel_file)
        
        # Save to Excel
        df.to_excel(excel_path, index=False)
        print(f'Results saved to {excel_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='Eureka')
    parser.add_argument('--checkpoint', type=str, default='/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/zhangkaipeng-24043/zhoupengfei/K12_onlinefilter_Episode10')
    parser.add_argument('--data-root', type=str, default='/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/zhangkaipeng-24043/zkp/MDK12mini-medium', help='Directory containing TSV files to process')
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    args.out_dir = os.path.join(args.out_dir, args.model_name)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=2,
        # max_model_len=32768,
        limit_mm_per_prompt={"image": 1},
        # mm_processor_kwargs={"max_dynamic_patch": 6},
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None
    
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] max_num: {args.max_num}')
    print(f'[test] data_root: {args.data_root}')

    evaluate_chat_model()
