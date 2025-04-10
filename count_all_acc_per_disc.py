import pandas as pd 
import pickle
import numpy as np
from collections import defaultdict
import os

# Subject list
disciplines = [
    'GSMA', 'MSMA', 'MSPH', 'MSCH', 'MSGE', 'MSBI', 'MSIT',
    'HSMA', 'HSPH', 'HSCH', 'HSGE', 'HSBI', 'HSIT'
]

# Task type list
task_types = [
    'fill_in_the_blank', 'calculation', 'multiple_answer_choice',
    'single_answer_choice', 'true_false'

]

model = 'Eureka'
# Modality list
modalities = ['multi_modal', 'single_modal']

# Input your base root path
base_path = 'XX'
result_path = f'{base_path}/results/{model}'

# Create output directory
output_dir = f'./results_{model}_{result_path.split("/")[-2]}'
os.makedirs(output_dir, exist_ok=True)

# Select suffix rule
def get_file_suffix(task_type):
    if task_type == 'single_answer_choice':
        return '_openai_result.xlsx'
    elif task_type == 'true_false':
        return '_auxmatch.xlsx'
    else:
        return '_gpt-4o.xlsx'

# K12 score calculation
def k12_score(data):
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    cate_list = []
    for _, item in data.iterrows():
        cate = item['year']
        cate2 = item['difficulty_level']
        cate_list.extend([cate, cate2])
        grade = float(item['score'])
        tot['Overall'] += 1
        tot[cate] += 1
        tot[cate2] += 1
        score['Overall'] += grade
        score[cate] += grade
        score[cate2] += grade

    res = {'Category': [], 'tot': [], 'acc': []}
    for v in set(cate_list + ['Overall']):
        res['Category'].append(v)
        res['tot'].append(tot[v])
        res['acc'].append(score[v] / tot[v] * 100)
    return pd.DataFrame(res)

# true_false score calculation
def default_rating(data):
    # Filter out rows where extracted equals Unknown, and only keep rows where prediction contains yes/no/true/false
    filtered_data = data[data['extracted'] != 'Unknown']
    
    # Further filter, only keep rows where prediction contains yes/no/true/false/correct/incorrect
    yes_no_pattern = filtered_data['extracted'].str.lower().str.contains('yes|no|true|false|correct|incorrect')
    filtered_data = filtered_data[yes_no_pattern]
    
    print(f"Number of records after filtering: {len(filtered_data)}")
    print(f"Average value of score column: {np.nanmean(filtered_data['score'])}")
    
    res = {'Overall': np.nanmean(filtered_data['score']) * 100, 'true_false': np.nanmean(filtered_data['score']) * 100}
    if 'category' in data.columns:
        for c in sorted(set(filtered_data['category'])):
            sub = filtered_data[filtered_data['category'] == c]
            res[c] = np.nanmean(sub['score']) * 100
    if 'l2-category' in data.columns:
        for c in sorted(set(filtered_data['l2-category'])):
            sub = filtered_data[filtered_data['l2-category'] == c]
            res[c] = np.nanmean(sub['score']) * 100
    return pd.DataFrame([res])

# single_answer_choice score calculation
def report_acc(df):
    from collections import defaultdict
    res = defaultdict(list)
    if 'split' in df:
        res['split'] = list(set(df['split']))
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'l2-category', 'category']:
        if group is None:
            res['Overall'] = [np.nanmean(df[df['split'] == sp]['hit']) for sp in res['split']]
        elif group in df.columns:
            for ab in sorted(set(df[group])):
                sub_df = df[df[group] == ab]
                res[ab] = [np.nanmean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
    return pd.DataFrame(res)

# Grouped by major categories, such as MA, PH, CH, BI, GE, IT
grouped_disciplines = {
    'MA': ['GSMA', 'MSMA', 'HSMA'],
    'PH': ['MSPH', 'HSPH'],
    'CH': ['MSCH', 'HSCH'],
    'BI': ['MSBI', 'HSBI'],
    'GE': ['MSGE', 'HSGE'],
    'IT': ['MSIT', 'HSIT']
}

# Batch process all disciplines, task types and modalities
all_scores = []  # Store all score results for calculating the total average
subject_scores = {subject: [] for subject in disciplines}  # Store score results for each subject

for subject in disciplines:
    # If you want to skip certain disciplines, you can add a condition here
    # if subject == 'MSMA':
    #     continue

    # Create separate output directory for each subject
    subject_output_dir = f'{output_dir}/{subject}_results'
    os.makedirs(subject_output_dir, exist_ok=True)
    
    if not os.path.exists(result_path):
        print(f'Directory does not exist, skipping: {result_path}')
        continue
        
    for task_type in task_types:
        file_suffix = get_file_suffix(task_type)
        
        for modality in modalities:
            # Find files matching current subject, task_type and modality
            file_pattern = f'{model}_{subject}_{task_type}_{modality}'
            matching_files = [f for f in os.listdir(result_path) if file_pattern in f and f.endswith(file_suffix)]
            
            if not matching_files:
                print(f'No matching file found: {file_pattern}*{file_suffix}')
                continue
                
            result_file = os.path.join(result_path, matching_files[0])
                
            output_file = f'{subject_output_dir}/{subject}_{task_type}_{modality}.xlsx'
            score_output_file = f'{subject_output_dir}/{subject}_{task_type}_{modality}_score.csv'

            try:
                # Read result file for evaluation
                total_df = pd.read_excel(result_file)
                
                # Save results as a new Excel file
                total_df.to_excel(output_file, index=False)
                print(f'Saved as {output_file} file.')

                # Choose scoring method based on task type and collect total score
                if task_type in ['calculation', 'fill_in_the_blank', 'multiple_answer_choice']:
                    score_df = k12_score(total_df)
                    # Get Overall accuracy
                    overall_acc = score_df[score_df['Category'] == 'Overall']['acc'].values[0]
                    all_scores.append(overall_acc)
                    subject_scores[subject].append(overall_acc)

                elif task_type == 'true_false':
                    score_df = default_rating(total_df)
                    overall_acc = score_df['Overall'].values[0]
                    all_scores.append(overall_acc)
                    subject_scores[subject].append(overall_acc)

                elif task_type == 'single_answer_choice':
                    score_df = report_acc(total_df)
                    overall_acc = score_df['Overall'].values[0] * 100  # Convert to percentage
                    all_scores.append(overall_acc)
                    subject_scores[subject].append(overall_acc)

                else:
                    raise ValueError(f'Unknown task type: {task_type}')

                score_df.to_csv(score_output_file, index=False)
                print(f'Score calculation completed, saved as {score_output_file} file.')

            except Exception as e:
                print(f'Error processing file {result_file}: {str(e)}')

# Calculate total average accuracy
if all_scores:
    average_acc = np.nanmean(all_scores)
    print(f'Average accuracy for all tasks: {average_acc:.2f}%')
    
    # Calculate average accuracy for each subject
    subject_avg = {}
    for subject, scores in subject_scores.items():
        if scores:
            subject_avg[subject] = np.nanmean(scores)
            print(f'Average accuracy for {subject} subject: {subject_avg[subject]:.2f}%')
    
    # Save total average accuracy to file
    result_dict = {'Average_Accuracy': [average_acc]}
    
    # Save average accuracy for each subject to file
    subject_result_dict = {subject: [acc] for subject, acc in subject_avg.items()}
    pd.DataFrame(subject_result_dict).to_csv(os.path.join(output_dir, 'subject_average_accuracy.csv'), index=False)
    print(f'Subject average accuracies have been saved to {os.path.join(output_dir, "subject_average_accuracy.csv")} file')
    
    # Calculate average accuracy for six major categories
    grouped_subject_avg = {}
    for group_name, sub_list in grouped_disciplines.items():
        # Collect accuracies of sub-disciplines in the current category
        sub_averages = []
        for sub in sub_list:
            if sub in subject_avg:
                sub_averages.append(subject_avg[sub])
        # If there is data for sub-disciplines in this category, calculate the average
        if sub_averages:
            grouped_subject_avg[group_name] = np.nanmean(sub_averages)
            print(f'Average accuracy for {group_name} category: {grouped_subject_avg[group_name]:.2f}%')
    
    # Save "six major disciplines" average accuracy to CSV
    if grouped_subject_avg:
        df_data = {
            group_name: [acc] 
            for group_name, acc in grouped_subject_avg.items()
        }
        out_path = os.path.join(output_dir, 'grouped_discipline_average_accuracy.csv')
        pd.DataFrame(df_data).to_csv(out_path, index=False)
        print(f'Six major disciplines (MA, PH, CH, BI, GE, IT) average accuracies have been saved to {out_path} file')

    pd.DataFrame(result_dict).to_csv(os.path.join(output_dir, 'average_accuracy.csv'), index=False)
    print(f'Total average accuracy has been saved to {os.path.join(output_dir, "average_accuracy.csv")} file')

else:
    print('No valid score data found, cannot calculate average accuracy')
