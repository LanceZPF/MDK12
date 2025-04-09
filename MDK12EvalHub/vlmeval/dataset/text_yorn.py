from ..smp import *
from ..utils import *
from .text_base import TextBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
import os.path as osp
import os

class TextYORNDataset(TextBaseDataset):

    TYPE = 'Y/N'

    DATASET_URL = {
    }

    DATASET_MD5 = {
    }

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.yorn import YOrN_Extraction, YOrN_auxeval
        from .utils.yorn import default_rating

        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        storage = eval_file.replace('.xlsx', '_auxmatch.xlsx')
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            ans_map = {k: YOrN_Extraction(v) for k, v in zip(data['index'], data['prediction'])}
            if osp.exists(tmp_file):
                tmp = load(tmp_file)
                for k in tmp:
                    if ans_map[k] == 'Unknown' and tmp[k] != 'Unknown':
                        ans_map[k] = tmp[k]

            data['extracted'] = [ans_map[x] for x in data['index']]
            unknown = data[data['extracted'] == 'Unknown']

            model = judge_kwargs.get('model', 'exact_matching')
            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                model = None
                warnings.warn('OPENAI_API_KEY is not working properly, will use exact matching for evaluation')

            if model is not None:
                lt = len(unknown)
                lines = [unknown.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = list(unknown['index'])
                if len(tups):
                    res = track_progress_rich(
                        YOrN_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            data['extracted'] = [str(ans_map[x]).strip() for x in data['index']]
            dump(data, storage)
        
        data = load(storage)

        data['answer'] = [str(x) for x in data['answer']]

        data['answer'] = data['answer'].map({
            'True': 'Yes', 
            'False': 'No',
            'true': 'Yes',  # 添加小写情况
            'false': 'No',
            'TRUE': 'Yes',  # 添加大写情况
            'FALSE': 'No',
            'yes': 'Yes',   # 已经是Yes/No的情况
            'no': 'No',
            'Yes': 'Yes',
            'No': 'No'
        })

        data['score'] = (data['answer'] == data['extracted'])
        dump(data, storage)

        score = default_rating(storage)

        score_tgt = eval_file.replace('.xlsx', '_score.csv')
        dump(score, score_tgt)
        return score
