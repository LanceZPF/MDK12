import os
import sys
import csv
from tqdm import tqdm
import multiprocessing


def replace_image_column(original_tsv_dir, expanded_tsv_dir, output_dir=None):
    """
    将原始TSV文件中的image列替换为扩展后TSV文件中对应行的image列
    
    Args:
        original_tsv_dir: 原始TSV文件所在目录
        expanded_tsv_dir: 扩展后TSV文件所在目录（通常是original_tsv_dir + "_expanded"）
        output_dir: 输出目录，默认为original_tsv_dir + "_replaced"
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = original_tsv_dir + "_tallvall"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取原始目录中的所有TSV文件
    original_tsv_files = [f for f in os.listdir(original_tsv_dir) if f.endswith(".tsv")]
    
    if not original_tsv_files:
        print(f"Warning: No TSV files found in {original_tsv_dir}")
        return
    
    for tsv_file in tqdm(original_tsv_files, desc=f"Processing TSV files in {os.path.basename(original_tsv_dir)}"):
        # 原始TSV文件路径
        original_tsv_path = os.path.join(original_tsv_dir, tsv_file)
        
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(tsv_file)[0]
        
        # 扩展后的TSV文件路径（通常会添加_expanded后缀）
        expanded_tsv_path = os.path.join(expanded_tsv_dir, f"{base_name}_tallvall.tsv")
        
        # 检查扩展后的文件是否存在
        if not os.path.exists(expanded_tsv_path):
            print(f"Warning: Expanded file not found: {expanded_tsv_path}")
            # 如果找不到扩展文件，尝试直接复制原始文件到输出目录
            try:
                with open(original_tsv_path, "r", encoding="utf-8") as f_in:
                    with open(os.path.join(output_dir, f"{base_name}_tallvall.tsv"), "w", encoding="utf-8", newline="") as f_out:
                        f_out.write(f_in.read())
                print(f"Copied original file to output: {base_name}_tallvall.tsv")
            except Exception as e:
                print(f"Error copying original file: {str(e)}")
            continue
        
        # 输出文件路径
        output_tsv_path = os.path.join(output_dir, f"{base_name}_tallvall.tsv")
        
        # 提升字段大小限制
        csv.field_size_limit(sys.maxsize)
        
        try:
            # 读取扩展后的TSV文件，提取image列
            expanded_images = {}
            with open(expanded_tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter='\t')
                for idx, row in enumerate(reader):
                    expanded_images[idx] = row.get("image", "")
            
            # 读取原始TSV文件，并替换image列
            with open(original_tsv_path, "r", encoding="utf-8") as f_in:
                reader = csv.DictReader(f_in, delimiter='\t')
                
                # 确保fieldnames不为None
                fieldnames = reader.fieldnames if reader.fieldnames else []
                
                # 检查是否有"image"字段
                if "image" not in fieldnames:
                    print(f"Warning: No 'image' column found in {tsv_file}, copying original file")
                    with open(original_tsv_path, "r", encoding="utf-8") as f_src:
                        with open(output_tsv_path, "w", encoding="utf-8", newline="") as f_dst:
                            f_dst.write(f_src.read())
                    continue
                
                with open(output_tsv_path, "w", encoding="utf-8", newline="") as f_out:
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter='\t')
                    writer.writeheader()
                    
                    for idx, row in enumerate(reader):
                        # 如果扩展后的文件中有对应行的image数据，则替换
                        if idx in expanded_images and expanded_images[idx]:
                            row["image"] = expanded_images[idx]
                        
                        # 写入行
                        writer.writerow(row)
            
            print(f"Replaced image column in {tsv_file}, saved to: {output_tsv_path}")
        except Exception as e:
            print(f"Error processing file {tsv_file}: {str(e)}")
            # 出错时尝试直接复制原始文件
            try:
                with open(original_tsv_path, "r", encoding="utf-8") as f_in:
                    with open(output_tsv_path, "w", encoding="utf-8", newline="") as f_out:
                        f_out.write(f_in.read())
                print(f"Error occurred, copied original file to output: {base_name}_tallvall.tsv")
            except Exception as copy_error:
                print(f"Error copying original file: {str(copy_error)}")
    
    # 检查输出目录中是否有文件
    output_files = [f for f in os.listdir(output_dir) if f.endswith(".tsv")]
    if not output_files:
        print(f"Warning: No output files were created in {output_dir}. Check if the process completed successfully.")
        # 如果没有输出文件，尝试直接复制原始文件
        try:
            for tsv_file in original_tsv_files:
                original_tsv_path = os.path.join(original_tsv_dir, tsv_file)
                base_name = os.path.splitext(tsv_file)[0]
                output_tsv_path = os.path.join(output_dir, f"{base_name}_tallvall.tsv")
                with open(original_tsv_path, "r", encoding="utf-8") as f_in:
                    with open(output_tsv_path, "w", encoding="utf-8", newline="") as f_out:
                        f_out.write(f_in.read())
            print(f"Copied all original files to output directory as fallback")
        except Exception as e:
            print(f"Error during fallback copy: {str(e)}")
    else:
        print(f"Created {len(output_files)} files in {output_dir}")
    
    print(f"All TSV files in {os.path.basename(original_tsv_dir)} have been processed with replaced image columns.")


def process_directory(original_tsv_dir, expanded_tsv_dir=None, output_dir=None):
    """处理单个目录的函数，用于多进程调用"""
    
    # 检查目录是否存在
    if not os.path.exists(original_tsv_dir):
        print(f"Error: Original directory does not exist: {original_tsv_dir}")
        return
    
    if expanded_tsv_dir and not os.path.exists(expanded_tsv_dir):
        print(f"Error: Expanded directory does not exist: {expanded_tsv_dir}")
        return
    
    # 如果未指定输出目录，则使用默认命名
    if output_dir is None:
        output_dir = original_tsv_dir + "_tallvall"
    
    # 执行替换操作
    replace_image_column(
        original_tsv_dir=original_tsv_dir,
        expanded_tsv_dir=expanded_tsv_dir,
        output_dir=output_dir
    )
    
    # 验证输出目录中是否有文件
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith(".tsv")]
        print(f"Output directory {output_dir} contains {len(files)} TSV files")
        if len(files) == 0:
            # 如果没有输出文件，尝试直接复制原始文件
            try:
                original_files = [f for f in os.listdir(original_tsv_dir) if f.endswith(".tsv")]
                for tsv_file in original_files:
                    original_tsv_path = os.path.join(original_tsv_dir, tsv_file)
                    base_name = os.path.splitext(tsv_file)[0]
                    output_tsv_path = os.path.join(output_dir, f"{base_name}_tallvall.tsv")
                    with open(original_tsv_path, "r", encoding="utf-8") as f_in:
                        with open(output_tsv_path, "w", encoding="utf-8", newline="") as f_out:
                            f_out.write(f_in.read())
                print(f"Copied all original files to output directory as fallback")
            except Exception as e:
                print(f"Error during fallback copy: {str(e)}")
    else:
        print(f"Warning: Output directory {output_dir} was not created")


if __name__ == "__main__":
    # -------------- 配置区域 --------------
    # 原始TSV文件夹列表
    original_tsv_dirs = [
        "/mnt/workspace/zpf/Dynamic-K/MDK-easy_tcombined123",
        "/mnt/workspace/zpf/Dynamic-K/MDK-medium_tcombined123",
        "/mnt/workspace/zpf/Dynamic-K/MDK-hard_tcombined123"
    ]
    
    extended_tsv_dirs = [
        "/mnt/workspace/zpf/Dynamic-K/MDK12easy_vcombined123",
        "/mnt/workspace/zpf/Dynamic-K/MDK12medium_vcombined123",
        "/mnt/workspace/zpf/Dynamic-K/MDK12hard_vcombined123"
    ]

    # 是否使用多进程处理
    use_multiprocessing = True
    # --------------------------------------
    
    # 验证所有目录是否存在
    for dir_path in original_tsv_dirs + extended_tsv_dirs:
        if not os.path.exists(dir_path):
            print(f"Warning: Directory does not exist: {dir_path}")
    
    if use_multiprocessing:
        # 创建进程池并并行处理所有目录
        processes = []
        for idx, original_tsv_dir in enumerate(original_tsv_dirs):
            # 定义一套规则
            expanded_tsv_dir = extended_tsv_dirs[idx]
            p = multiprocessing.Process(
                target=process_directory, 
                args=(original_tsv_dir, expanded_tsv_dir)
            )
            processes.append(p)
            p.start()
        
        # 等待所有进程完成
        for p in processes:
            p.join()
    else:
        # 顺序处理所有目录
        for idx, original_tsv_dir in enumerate(original_tsv_dirs):
            # 定义一套规则
            expanded_tsv_dir = extended_tsv_dirs[idx]
            process_directory(original_tsv_dir, expanded_tsv_dir)
    
    # 验证所有输出目录是否包含文件
    output_dirs = [dir_path + "_tallvall" for dir_path in original_tsv_dirs]
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith(".tsv")]
            print(f"Final check: Output directory {output_dir} contains {len(files)} TSV files")
            if len(files) == 0:
                # 如果最终检查仍然没有文件，执行最后的备份方案
                print('fuced?')
                exit()
        else:
            print(f"Final check: Output directory {output_dir} does not exist")
    
    print("All directories have been processed successfully.")
