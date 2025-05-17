import os
import sys
# 若需要指定 GPU，请在执行脚本前或此处解开注释进行设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import sys
import numpy as np
import random
import multiprocessing
from main_image_expansion import ImageExpansion
from main_color_transfer import ColorTransfer


class CombinedImageProcessor:
    def __init__(self, random_seed=0):
        self.random_seed = random_seed
        # 设置随机种子以保证结果可复现
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 初始化两个图像处理器
        self.image_expansion = ImageExpansion(random_seed)
        self.color_transfer = ColorTransfer(random_seed)
    
    def apply_combined12_processing(self, image):
        """依次应用视野扩展和颜色转换处理"""
        # 第一步：应用视野扩展
        expanded_image = self.image_expansion.apply_expansion(image)
        
        # 第二步：应用颜色转换
        processed_image = self.color_transfer.apply_color_transfer(expanded_image)
        
        return processed_image
    
    def process_batch(self, tsv_dir, transfer_dir=None, images_dir=None):
        """批量处理TSV文件中的图像，依次应用视野扩展和颜色转换"""
        
        # 设置输出目录
        if transfer_dir is None:
            transfer_dir = tsv_dir + "_combined12_processing"
        if images_dir is None:
            images_dir = os.path.join(transfer_dir, "images")
            
        os.makedirs(transfer_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # 遍历目标文件夹下的所有 .tsv 文件
        tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith(".tsv")]
        for tsv_file in tqdm(tsv_files, desc=f"Processing TSV files in {os.path.basename(tsv_dir)}"):
            tsv_path = os.path.join(tsv_dir, tsv_file)
            base_name = os.path.splitext(tsv_file)[0]

            # 提升字段大小限制
            csv.field_size_limit(sys.maxsize)

            # 读取原 TSV 内容
            with open(tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)

            # 更新后的 TSV 将写到新的文件中
            out_tsv_path = os.path.join(transfer_dir, f"{base_name}_combined12.tsv")
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

                    # 应用组合处理：先视野扩展，再颜色转换
                    processed_image = self.apply_combined12_processing(init_image)

                    # 保存处理后的图像到 images 文件夹
                    processed_image_name = f"combined_{base_name}_{idx}.jpg"
                    processed_image_path = os.path.join(images_dir, processed_image_name)
                    processed_image.save(processed_image_path)

                    # 将处理后的图像重新编码为 base64，并更新到当前 TSV 行
                    buffered = BytesIO()
                    processed_image.save(buffered, format="JPEG")
                    new_b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    row["image"] = new_b64_data

                    # 写入更新后的行
                    writer.writerow(row)

            print(f"Updated TSV saved to: {out_tsv_path}")

        print(f"All TSV files in {os.path.basename(tsv_dir)} have been processed and saved with new base64-encoded images.")


def process_directory(tsv_dir, random_seed=0):
    """处理单个目录的函数，用于多进程调用"""
    transfer_dir = tsv_dir + "_combined12_processing"
    images_dir = os.path.join(transfer_dir, "images")
    
    # 创建组合图像处理器实例
    combined_processor = CombinedImageProcessor(random_seed)
    
    # 执行批量处理
    combined_processor.process_batch(
        tsv_dir=tsv_dir,
        transfer_dir=transfer_dir,
        images_dir=images_dir
    )


if __name__ == "__main__":
    # -------------- 配置区域 --------------
    # 目标文件夹列表，内含 .tsv 文件
    tsv_dirs = [
        "/mnt/workspace/zpf/Dynamic-K/MDK12hard",
        "/mnt/workspace/zpf/Dynamic-K/MDK12medium",
        "/mnt/workspace/zpf/Dynamic-K/MDK12easy"
    ]
    
    # 随机种子设置
    random_seed = 0
    # --------------------------------------
    
    # 创建进程池并并行处理所有目录
    processes = []
    for tsv_dir in tsv_dirs:
        p = multiprocessing.Process(target=process_directory, args=(tsv_dir, random_seed))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("All directories have been processed successfully.")
