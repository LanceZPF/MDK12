from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


# Global flag to use vLLM for inference
USE_VLLM = os.environ.get('USE_VLLM', '0') == '1'


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant\'s output should start with <think> and end with </answer>."
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path

        # Set tensor_parallel_size to the number of available GPUs, or use the provided value as fallback
        try:
            num_gpus = torch.cuda.device_count()
            self.tensor_parallel_size = num_gpus if num_gpus > 0 else tensor_parallel_size
        except:
            self.tensor_parallel_size = tensor_parallel_size
        
        # MODEL_CLS = None  
        # if '2.5' in model_path:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        MODEL_CLS = Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path)
        # else:
        #     from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        #     MODEL_CLS = Qwen2VLForConditionalGeneration
        #     self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        if USE_VLLM:
            try:
                from vllm import LLM, SamplingParams
                from qwen_vl_utils import process_vision_info
                
                self.llm = LLM(
                    model=model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=self.tensor_parallel_size,
                    # limit_mm_per_prompt={"image": 1},
                    # rope_scaling={"type": "linear", "factor": 1.0}
                )
                self.sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=None, skip_special_tokens=False)
                self.process_vision_info = process_vision_info
                self.use_vllm = True
                print("Using vLLM for inference")
            except ImportError:
                warnings.warn("vLLM not found, falling back to standard inference. Install vLLM with 'pip install vllm'")
                self.use_vllm = False
        else:
            self.use_vllm = False
            
        if not self.use_vllm:
            gpu_mems = get_gpu_memory()
            max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
            assert max_gpu_mem > 0

            # If only one process and GPU memory is less than 40GB
            if '72b' in self.model_path.lower():
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='flash_attention_2'
                )
                self.model.eval()
            elif auto_split_flag():
                assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
                # Will Use All GPUs to run one model
                # self.model = MODEL_CLS.from_pretrained(
                #     model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                # )
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map='cpu'
                )
                self.model.cuda().eval()
            else:
                # self.model = MODEL_CLS.from_pretrained(
                #     model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
                # )
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map='cpu'
                )
                self.model.cuda().eval()

            torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self._generate_with_vllm(message, dataset)
        else:
            return self._generate_with_transformers(message, dataset)
            
    def _generate_with_vllm(self, message, dataset=None):
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': [
                                {"type": "text", "text": self.system_prompt
                                },]
                             })
        # print(message)
        # exit()
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')
            
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = self.process_vision_info(messages)
        
        # Check if there's image data
        if image_data:
            inputs = [{
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
            }]
        else:
            # Handle case when there's no image data
            inputs = [{
                "prompt": prompt,
                "multi_modal_data": None,
            }]
        
        model_outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        
        response = model_outputs[0].outputs[0].text
        
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]
                
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def _generate_with_transformers(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
