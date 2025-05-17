import os
import pandas as pd
import random

def random_sample_from_tsvs(target_folder, proportion):
    """
    Randomly sample rows from multimodal TSV files based on proportion.
    Args:
        target_folder: Folder containing TSV files
        n_samples: Target number of samples (used for output directory naming only)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join('/Users/lance/Library/CloudStorage/OneDrive-个人/xuekeMLLM/动态评测/Dynamic-K', "MDK12easy") 
    os.makedirs(output_dir, exist_ok=True)

    # Collect all rows from multimodal TSV files
    multimodal_files = {}
    total_rows = 0
    
    # First pass: identify multimodal files and count total rows
    for file in os.listdir(target_folder):
        if file.endswith('.tsv') and 'multi_modal' in file:
            file_path = os.path.join(target_folder, file)
            try:
                df = pd.read_csv(file_path, sep='\t')
                row_count = len(df)
                multimodal_files[file] = {'df': df, 'row_count': row_count}
                total_rows += row_count
                print(f"Found multimodal file: {file} with {row_count} rows")
            except Exception as e:
                print(f"Error reading file {file}: {e}")
    
    print(f"\nTotal rows in all multimodal files: {total_rows}")
    
    # Second pass: sample from each file based on proportion
    file_to_rows = {}
    total_sampled = 0
    
    for file, info in multimodal_files.items():
        df = info['df']
        row_count = info['row_count']
        
        # Calculate proportion of rows to sample from this file
        sample_size = int(row_count * proportion)
        
        if sample_size > 0:
            if sample_size > row_count:
                sample_size = row_count
                
            # Sample rows from this file
            sampled_indices = random.sample(range(row_count), sample_size)
            sampled_rows = df.iloc[sampled_indices]
            
            file_to_rows[file] = sampled_rows
            total_sampled += sample_size
            print(f"Sampled {sample_size} rows from {file} ({proportion:.2%} of total)")

    print(f"\nTotal sampled rows: {total_sampled} out of {total_rows}")
    
    # Add percentage calculation only if total_rows is not zero
    if total_rows > 0:
        print(f"Sampling percentage: {total_sampled/total_rows:.2%}")

    # Write sampled rows back to files
    for file, sampled_df in file_to_rows.items():
        if not sampled_df.empty:
            output_path = os.path.join(output_dir, file)
            sampled_df.to_csv(output_path, sep='\t', index=False)
            print(f"Wrote {len(sampled_df)} rows to {output_path}")

if __name__ == "__main__":
    target_folder = '/Users/lance/Downloads/MDK12mini-easy'  # Replace with actual path
    random_sample_from_tsvs(target_folder, proportion=0.5)



# tsv_dir = "/Users/lance/Downloads/MDK12mini-easy"       # 目标文件夹，内含 .tsv 文件
# os.makedirs(tsv_dir + "_transfer", exist_ok=True)
# transfer_dir = tsv_dir + "_transfer"  # 输出文件夹，用于保存更新后的 TSV
# images_dir = os.path.join(transfer_dir, "images")  # 用于保存原图和风格迁移后的图像