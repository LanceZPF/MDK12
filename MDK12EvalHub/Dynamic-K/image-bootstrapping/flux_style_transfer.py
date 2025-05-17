import os
import sys
# 若需要指定 GPU，请在执行脚本前或此处解开注释进行设置
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import csv
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import sys
import torch
from diffusers import FluxImg2ImgPipeline


class StyleTransfer:
    def __init__(self, model_id, device="cuda", random_seed=0):
        self.model_id = model_id
        self.device = device
        
        # 初始化 Flux 图像迁移管道
        self.i2i_pipe = FluxImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        ).to(device)
        
        # 用于可控随机性的随机数生成器
        self.generator = torch.Generator("cpu").manual_seed(random_seed)
    
    def process_batch(self, tsv_dir, transfer_dir=None, images_dir=None, 
                     prompt="A slightly stylized version of the original image with subtle art effects.",
                     strength_val=0.2, guidance_scale_val=3.5, 
                     num_inference_steps_val=50, max_sequence_length_val=512):
        """批量处理TSV文件中的图像进行风格迁移"""
        
        # 设置输出目录
        if transfer_dir is None:
            transfer_dir = tsv_dir + "_transfer"
        if images_dir is None:
            images_dir = os.path.join(transfer_dir, "images")
            
        os.makedirs(transfer_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # 遍历目标文件夹下的所有 .tsv 文件
        tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith(".tsv")]
        for tsv_file in tqdm(tsv_files, desc="Processing TSV files"):
            tsv_path = os.path.join(tsv_dir, tsv_file)
            base_name = os.path.splitext(tsv_file)[0]

            # 提升字段大小限制
            csv.field_size_limit(sys.maxsize)

            # 读取原 TSV 内容
            with open(tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)

            # 更新后的 TSV 将写到新的文件中
            out_tsv_path = os.path.join(transfer_dir, f"{base_name}_transfer.tsv")
            with open(out_tsv_path, "w", encoding="utf-8", newline="") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames, delimiter='\t')
                writer.writeheader()

                # 对每一行进行处理
                for idx, row in tqdm(enumerate(rows), desc="Processing lines", leave=False):
                    # 取出 TSV 中的 base64 图片数据
                    b64_data = row.get("image", "")
                    if not b64_data:
                        # 如果没有 image 字段或内容为空，则直接写回行
                        writer.writerow(row)
                        continue

                    # base64 解码 -> PIL Image
                    try:
                        image_bytes = base64.b64decode(b64_data)
                        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    except Exception as e:
                        print(f"Warning: Failed to decode base64 at line {idx} in {tsv_file}: {e}")
                        # 无法解码的情况，直接跳过或写回原行
                        writer.writerow(row)
                        continue

                    # 保存原图到 images 文件夹
                    original_image_name = f"original_{base_name}_{idx}.jpg"
                    original_image_path = os.path.join(images_dir, original_image_name)
                    init_image.save(original_image_path)

                    # 使用 Flux 进行轻微风格迁移
                    styled_image = self.i2i_pipe(
                        prompt=prompt,
                        image=init_image,
                        height=1024,
                        width=1024,
                        strength=strength_val,
                        guidance_scale=guidance_scale_val,
                        num_inference_steps=num_inference_steps_val,
                        max_sequence_length=max_sequence_length_val,
                        generator=self.generator
                    ).images[0]

                    # 保存迁移后的图像到 images 文件夹
                    styled_image_name = f"styled_{base_name}_{idx}.jpg"
                    styled_image_path = os.path.join(images_dir, styled_image_name)
                    styled_image.save(styled_image_path)

                    # 将迁移后的图像重新编码为 base64，并更新到当前 TSV 行
                    buffered = BytesIO()
                    styled_image.save(buffered, format="JPEG")
                    new_b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    row["image"] = new_b64_data

                    # 写入更新后的行
                    writer.writerow(row)

            print(f"Updated TSV saved to: {out_tsv_path}")

        print("All TSV files have been processed and saved with new base64-encoded images.")


if __name__ == "__main__":

    # -------------- 配置区域 --------------
    device = "cuda"
    model_id = "/mnt/workspace/zpf/.cache/FLUX.1-dev"

    tsv_dir = "/mnt/workspace/zpf/Dynamic-K/MDK12hard"       # 目标文件夹，内含 .tsv 文件
    transfer_dir = tsv_dir + "_transfer"  # 输出文件夹，用于保存更新后的 TSV
    images_dir = os.path.join(transfer_dir, "images")  # 用于保存原图和风格迁移后的图像

    # 轻微风格迁移的提示语
    prompt = "A slightly stylized version of the original image with subtle art effects."

    # 风格迁移的相关参数
    strength_val = 0.2
    guidance_scale_val = 3.5
    num_inference_steps_val = 50
    max_sequence_length_val = 512
    random_seed = 0
    # --------------------------------------

    

    # 创建风格迁移处理器实例
    style_transfer = StyleTransfer(model_id, device, random_seed)

    # 执行批量处理
    style_transfer.process_batch(
        tsv_dir=tsv_dir,
        transfer_dir=transfer_dir,
        images_dir=images_dir,
        prompt=prompt,
        strength_val=strength_val,
        guidance_scale_val=guidance_scale_val,
        num_inference_steps_val=num_inference_steps_val,
        max_sequence_length_val=max_sequence_length_val
    )
