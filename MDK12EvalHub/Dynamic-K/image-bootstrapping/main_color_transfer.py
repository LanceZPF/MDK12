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


class ColorTransfer:
    def __init__(self, random_seed=0):
        self.random_seed = random_seed
        # 设置随机种子以保证结果可复现
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def apply_color_transfer(self, image):
        """对图像应用取反色和椒盐噪声效果"""
        # 转换为numpy数组以便处理
        img_array = np.array(image)
        
        # 取反色 (255 - 原始值)
        inverted_img = 255 - img_array
        # 添加基于泊松过程和马尔可夫链的椒盐噪声
        noise_prob = 0.01  # 基础噪声比例
        salt_vs_pepper = 0.5  # 白噪声与黑噪声的基础比例
        
        # 使用泊松过程确定噪声点的数量
        height, width = inverted_img.shape[:2]
        total_pixels = height * width
        lambda_param = noise_prob * total_pixels  # 泊松分布的λ参数
        noise_count = np.random.poisson(lambda_param)
        
        # 初始化马尔可夫链状态（0=无噪声, 1=盐噪声, 2=椒噪声）
        current_state = np.random.choice([0, 1, 2], p=[1-noise_prob, noise_prob*salt_vs_pepper, noise_prob*(1-salt_vs_pepper)])
        
        # 马尔可夫链转移矩阵
        # 行和列分别代表当前状态和下一状态：[无噪声, 盐噪声, 椒噪声]
        transition_matrix = np.array([
            [0.99, 0.005, 0.005],  # 从无噪声状态转移的概率
            [0.3, 0.6, 0.1],       # 从盐噪声状态转移的概率
            [0.3, 0.1, 0.6]        # 从椒噪声状态转移的概率
        ])
        
        # 创建噪声掩码和颜色映射
        salt_mask = np.zeros((height, width), dtype=bool)
        pepper_mask = np.zeros((height, width), dtype=bool)
        
        # 为盐噪声和椒噪声生成随机颜色映射
        salt_colors = {}  # 存储盐噪声位置及其对应的随机颜色
        pepper_colors = {}  # 存储椒噪声位置及其对应的随机颜色
        
        # 使用马尔可夫链生成空间相关的噪声
        for _ in range(noise_count):
            # 随机选择位置
            y, x = np.random.randint(0, height), np.random.randint(0, width)
            
            # 根据当前状态和转移矩阵确定下一状态
            current_state = np.random.choice([0, 1, 2], p=transition_matrix[current_state])
            
            # 根据状态应用噪声
            if current_state == 1:  # 盐噪声
                salt_mask[y, x] = True
                # 为这个位置生成随机亮色
                salt_colors[(y, x)] = np.random.randint(180, 256, size=3)
            elif current_state == 2:  # 椒噪声
                pepper_mask[y, x] = True
                # 为这个位置生成随机暗色
                pepper_colors[(y, x)] = np.random.randint(0, 76, size=3)
                
            # 有一定概率在周围像素也添加相同类型的噪声（模拟噪声的空间相关性）
            if np.random.random() < 0.3:  # 30%的概率扩散到邻近像素
                for dy, dx in [(0,1), (1,0), (0,-1), (-1,0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and np.random.random() < 0.7:
                        if current_state == 1:
                            salt_mask[ny, nx] = True
                            # 使用相似的颜色（添加小的随机变化）
                            if (y, x) in salt_colors:
                                base_color = salt_colors[(y, x)]
                                variation = np.random.randint(-20, 21, size=3)
                                salt_colors[(ny, nx)] = np.clip(base_color + variation, 180, 255)
                            else:
                                salt_colors[(ny, nx)] = np.random.randint(180, 256, size=3)
                        elif current_state == 2:
                            pepper_mask[ny, nx] = True
                            # 使用相似的颜色（添加小的随机变化）
                            if (y, x) in pepper_colors:
                                base_color = pepper_colors[(y, x)]
                                variation = np.random.randint(-20, 21, size=3)
                                pepper_colors[(ny, nx)] = np.clip(base_color + variation, 0, 75)
                            else:
                                pepper_colors[(ny, nx)] = np.random.randint(0, 76, size=3)
        
        # 应用噪声和颜色
        for (y, x), color in salt_colors.items():
            if salt_mask[y, x]:  # 确保位置仍在掩码中
                inverted_img[y, x] = color
                
        for (y, x), color in pepper_colors.items():
            if pepper_mask[y, x]:  # 确保位置仍在掩码中
                inverted_img[y, x] = color
        
        # 转回PIL图像
        processed_image = Image.fromarray(inverted_img.astype(np.uint8))
        return processed_image
    def process_batch(self, tsv_dir, transfer_dir=None, images_dir=None):
        """批量处理TSV文件中的图像进行颜色转换"""
        
        # 设置输出目录
        if transfer_dir is None:
            transfer_dir = tsv_dir + "_color_transfer"
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
            out_tsv_path = os.path.join(transfer_dir, f"{base_name}_color_transfer.tsv")
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

                    # 应用颜色转换（取反色和椒盐噪声）
                    processed_image = self.apply_color_transfer(init_image)

                    # 保存处理后的图像到 images 文件夹
                    processed_image_name = f"processed_{base_name}_{idx}.jpg"
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
    transfer_dir = tsv_dir + "_color_transfer"
    images_dir = os.path.join(transfer_dir, "images")
    
    # 创建颜色转换处理器实例
    color_transfer = ColorTransfer(random_seed)
    
    # 执行批量处理
    color_transfer.process_batch(
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
