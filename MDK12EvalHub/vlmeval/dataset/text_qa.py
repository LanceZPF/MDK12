import os
import re
import pandas as pd
from .text_base import TextBaseDataset
from ..smp import *
from ..utils import track_progress_rich, can_infer
from .utils import build_judge, DEBUG_MESSAGE

import numpy as np
import multiprocessing as mp
from functools import partial

# pip install latex2sympy2
from latex2sympy2 import latex2sympy
FAIL_MSG = 'Failed to obtain answer via API.'

class TextQADataset(TextBaseDataset):
    TYPE = 'QA'
    DATASET_URL = {}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
    
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vqa_eval import hit_calculate, process_line

        data = load(eval_file)
        
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # 如果question和answer都同时包含"(1)"和"(2)"，则截取answer中"(2)"之前的字符串
        if 'question' in data:
            for idx, row in data.iterrows():
                if all(token in row['question'] for token in ["(1)", "(2)"]) and all(token in row['answer'] for token in ["(1)", "(2)"]):
                    data.at[idx, 'answer'] = row['answer'].split("(2)")[0]


        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        
        # Use accuracy method for text QA evaluation
        res = pool.map(partial(process_line, method='accuracy'), lines)
        hit = hit_calculate(res, self.dataset_name)
        
        ret = dict()
        if 'split' in data:
            splits = set(data['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                hit = hit_calculate(sub, self.dataset_name)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = hit_calculate(sub, self.dataset_name)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            # 这个category可能要根据数据集情况改/增加一下键名
            if 'category' in data:
                cates = list(set(data['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    hit = hit_calculate(sub, self.dataset_name)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)

        return ret


class MathBench(TextQADataset):
    DATASET_TYPE = 'QA'
    MODALITY = 'TEXT'

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        tot = defaultdict(lambda: 0)
        fetch = defaultdict(lambda: 0) 
        hit = defaultdict(lambda: 0)
        lt = len(data)
        skill_list = []

        for i in range(lt):
            item = data.iloc[i]
            cate = item['task'] if 'task' in item else 'default'
            tot['Overall'] += 1
            
            try:
                if 'skills' in item:
                    try:
                        skills = eval(item['skills'])
                    except SyntaxError:
                        skills = [item['skills']]
                    for skill in skills:
                        if skill not in skill_list:
                            skill_list.append(skill)
                        tot[skill] += 1
                tot[cate] += 1

                if item.get('question_type') == 'multi_choice':
                    choices = list_to_dict(eval(item['choices']))
                    pred = can_infer(item['prediction'], choices)
                    correct = pred == item['answer_option']
                else:
                    if item.get('answer_type') == 'integer':
                        pred = int(item['prediction'])
                        ans = int(item['answer'])
                    elif item.get('answer_type') == 'float':
                        pred = float(item['prediction'])
                        ans = float(item['answer'])
                    else:
                        pred = str(item['prediction'])
                        ans = str(item['answer'])
                    correct = pred == ans

                if correct:
                    hit['Overall'] += 1
                    hit[cate] += 1
                    if 'skills' in item:
                        for skill in skills:
                            hit[skill] += 1

            except (ValueError, TypeError):
                pass

        res = defaultdict(list)
        for k in tot.keys():
            res['Task&Skill'].append(k)
            res['tot'].append(tot[k])
            res['hit'].append(hit[k])
            res['acc'].append(hit[k] / tot[k] * 100)

        results_df = pd.DataFrame(res)
        score_file = eval_file.replace('.json', '_acc.csv')
        dump(results_df, score_file)
        return results_df

    @classmethod
    def supported_datasets(cls):
        return ['MathBench']


class UnifiedTextQADataset(TextBaseDataset):
    TYPE = 'QA'
    DATASET_URL = {}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        return load(data_path)

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        judge_kwargs['model'] = 'gpt-4o'
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{judge_kwargs["model"]}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{judge_kwargs["model"]}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        judge_model2 = build_judge(max_tokens=4, **judge_kwargs)
        assert judge_model2.working(), ('CustomVQA evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('CustomVQA evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, judge_model2, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    Text_eval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    if k not in ans:
                        print(f"Warning: Index {k} not found in saved results")
                        continue
                    if ans[k]['log'] != v['log'] or ans[k]['res'] != v['res']:
                        print(f"Warning: Inconsistency found for index {k}")
                        # Update with the latest result to ensure consistency
                        ans[k] = v

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            dump(data, storage)

        score = k12_score(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


def Text_eval(model, judge_model2, line):
    import time

    # First try prefetch check
    prompt = build_textqa_gpt4_prompt(line)

    log = ''
    if post_check(line, prefetch=True):
        res = post_check(line, prefetch=True)
        if post_check(line, prefetch=False):
            score = 1.0
        else:
            score = 0.0
    else:
        # If prefetch fails, try API calls
        retry = 5
        for i in range(retry):
            prediction = line['prediction']
            res = model.generate(prompt, temperature=i * 0.5)
            time.sleep(0.1)

            if FAIL_MSG in res:
                log += f'Try {i}: output is {prediction}, failed to parse.\n'
                score = 0.0
            else:
                # Store the generated response before checking
                line = line.copy()  # Create a copy to avoid modifying original
                line['res'] = res   # Add the 'res' key before checking
                if post_check(line, prefetch=False):
                    score = 1.0
                else:
                    score = 0.0
                log += 'Succeed'
                break
        else:
            # All retries failed
            log += 'All 5 retries failed.\n'
            res = prediction
            score = 0.0
    
    if score == 0.0:
        judge_prompt = build_k12text_gpt4_prompt(line, log)
        log = ''
        retry = 3
        for i in range(retry):
            output = judge_model2.generate(judge_prompt, temperature=i * 0.5)
            score = float_cvt(output)
            if score is None:
                log += f'Try {i}: output is {output}, failed to score.\n'
                time.sleep(0.1)
                score = 0.0
            elif score < 0 or score > 1:
                log += f'Try {i}: output is {output}, invalid score: {score}.\n'
                time.sleep(0.1)
                score = 0.0
            else:
                log = 'Succeed'
                break
    
    return dict(log=log, res=res, score=score)

def build_textqa_gpt4_prompt(line):
    task_description = """
        Please read the following example.
        Then extract the answer from the model response and type it at the end of the prompt.\n
        """
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def post_check(line, prefetch=False):

    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']

    res = str(response)
    ans = str(ans)

    if is_equal(res, ans):
        return res if prefetch else True
    elif res in ans and len(ans.replace(res,'')) <= 5 \
        or ans in res and len(res.replace(ans,'')) <= 5:
        return res if prefetch else True
    else:
        return False
    

def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""
    example_6 = """
Question: If $X(k)$ is the N-point DFT of a sequence $x(n)$, then circular time shift property is that N-point DFT of $x((n-I))_N$ is $X(k) e^{-j 2 \pi k \mid / N}$. Is it true?\n
Model response: Yes, the statement is true. The circular time shift property of the N-point Discrete Fourier Transform (DFT) states that if \( X(k) \) is the N-point DFT of a sequence \( x(n) \), then the N-point DFT of the sequence \( x((n-I)_N) \) (where \( (n-I)_N \) denotes the circular shift of \( n \) by \( I \) modulo \( N \)) is given by \( X(k) e^{-j 2 \pi k I / N} \). This property reflects how the DFT transforms a circular shift in the time domain into a multiplication by a complex exponential in the frequency domain.\n
Extracted answer: Yes
"""

    example_7 = """
Question: Is the set of 3 * 3 matrices in reduced row-echelon form a subspace of R^{3 * 3}?\n
Model response: No, the set of 3 * 3 matrices in reduced row-echelon form is not a subspace of R^{3 * 3}.\n
Extracted answer: No
"""

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

def is_equal(asw: str, gt_asw: str) -> bool:
    if not isinstance(asw, str) != str or not isinstance(gt_asw, str):
        print('Warning: input is not string')
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-8:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-8:
            return True
        if abs(a - b) < 1e-8:
            return True
    except:
        pass
    return False

def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}



def build_k12text_gpt4_prompt(line, log):
    question = line['question']
    gt = str(line['answer'])
    # prediction = str(line['prediction'])
    prediction = line['res'] if log == 'Succeed' else str(line['prediction'])
    # prompt = """
    #         Compare the ground truth and prediction from AI models, to give a correctness score for the prediction.
    #         The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
    #         Just complete the last space of the correctness score.
    #         Question | Ground truth | Prediction | Correctness
    #         --- | --- | --- | ---
    #         What is x in the equation? | -1 | x = 3 | 0.0
    #         What is x in the equation? | -1 ; -5 | -1 | 0.5
    #         What is x in the equation? | -1 ; -5 | 3 ; -5 | 0.5
    #         What is x in the equation? | -1 ; -5 | x = -1 ; -5 | 1.0
    #         """
    prompt = """
            Compare the ground truth and prediction from AI models, to give a correctness score for the prediction.
                    The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
                    Just complete the last space of the correctness score.
                    Question | Ground truth | Prediction | Correctness
                    --- | --- | --- | ---
                What is x in the equation? | -1 | x = 3 | 0.0
                What is x in the equation? | -1 ; -5 | -1 | 0.5
                What is x in the equation? | -1 ; -5 | -5 | 0.5
                What is x in the equation? | -1 ; -5 | -1 ; 5 | 0.5
                What is x in the equation? | -1 ; -5 | 3 ; -5 | 0.5
                What is x in the equation? | -1 ; -5 | x = -1 ; -5 | 1.0
                There are three types of RNA: \\ul{　 　}. | Messenger RNA (mRNA), transfer RNA (tRNA), ribosomal RNA (rRNA) | [mRNA, tRNA, rRNA] | 1.0
                Set up the equation for calculation. (1) 75% of a number is 4.5 more than 60% of it. Find the number. (2) Twice a number is 3 less than \\(\\frac{1}{6}\\) of 54. Find the number. | (1) Solution: 4.5 ÷ (75% - 60%) = 4.5 ÷ 15% = 30. Answer: This number is 30. (2) Solution: (54 × \\(\\frac{1}{6}\\) - 3) ÷ 2 = (9 - 3) ÷ 2 = 6 ÷ 2 = 3. Answer: This number is 3. | [18, 3] | 0.5
                Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that Iceland is not very icy while Greenland isn't very green. | 0.5
                Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
            """
    gpt4_prompt = prompt + '\n' + ' | '.join([str(question), str(gt), str(prediction), ''])
    return gpt4_prompt

def float_cvt(s):
    try:
        return float(s)
    except ValueError:
        return None

def k12texteval(model, line):

    prompt = build_k12text_gpt4_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        output = model.generate(prompt, temperature=i * 0.5)
        score = float_cvt(output)
        if score is None:
            log += f'Try {i}: output is {output}, failed to parse.\n'
        elif score < 0 or score > 1:
            log += f'Try {i}: output is {output}, invalid score: {score}.\n'
        else:
            log += 'Succeed'
            return dict(log=log, score=[1.0, score])
    log += 'All 5 retries failed.\n'
    return dict(log=log, score=[0.0, 0.0])

def k12_score(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    lt = len(data)
    cate_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['year']
        cate2 = item['difficulty_level']
        if cate not in cate_list:
            cate_list.append(cate)
        if cate2 not in cate_list:
            cate_list.append(cate2)
        grade = float(item['score'])
        tot['Overall'] += 1
        tot[cate] += 1
        tot[cate2] += 1
        score['Overall'] += grade
        score[cate] += grade
        score[cate2] += grade

    res = defaultdict(list)
    cate_list.append('Overall')
    for v in cate_list:
        res['Category'].append(v)
        res['tot'].append(tot[v])
        res['acc'].append(score[v] / tot[v] * 100)
    res = pd.DataFrame(res)
    return res