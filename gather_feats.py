import tiny_model
from tiny_model import TinyModel
from datasets import load_dataset
import tokre
from utils import *

tokre.setup(tokenizer=tiny_model.tokenizer)
import json

MLP_DIR = 'mlp_map_test'
MLP_NAME = 'M0_S-2_R1_P0'
TOT_FEATS = 9662


import numpy as np
from utils import all_tok_ids
import os
from tqdm import tqdm
import ray
import time

ray.init(ignore_reinit_error=True)



@ray.remote(num_cpus=1)
class Actor:
    def __init__(self):
        dataset = load_dataset('noanabeshima/TinyModelTokIds', split='train[:13%]')
        self.all_tok_ids = torch.tensor(dataset['tok_ids'])
        tokre.setup(tokenizer=tiny_model.tokenizer, workspace='workspace')

    def process_feat(self, feat_idx):
        feat_acts = load_sparse_feat_acts(f'{MLP_DIR}/{MLP_NAME}/{feat_idx}.pt').to_sparse_coo()
        indices = feat_acts.indices()[:,:600]
        active_tok_ids = self.all_tok_ids[indices[0], indices[1]]

        uniq, counts = torch.unique(active_tok_ids, return_counts=True)
        
        num_unique = uniq.shape[0]
        topk_counts, _topk_indices = counts.topk(k=min(num_unique, 5))
        topk_ids = uniq[_topk_indices]

        pattern = "(" + "|".join([tokre.escape(tok) for tok in tiny_model.raw_toks[topk_ids]]) + ")"
        try:
            
            pos_r_squared = self.test_pattern(MLP_NAME, feat_idx, pattern + '[pos]')

            data = {
                'pattern': pattern,
                'pos_r_squared': pos_r_squared.item()
            }

            if pos_r_squared > 0.98:
                r_squared = self.test_pattern(MLP_NAME, feat_idx, pattern)
                data['r_squared'] = r_squared.item()
            
            
            output_dir = os.path.join('tokenset_feats', MLP_NAME)
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f'{feat_idx}.json')
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(pattern)
            print(f"An exception occurred: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            raise e

    def test_pattern(self, mlp_name, feat_idx, pattern, train_amt=0.8):
        acts = load_sparse_feat_acts(f'mlp_map_test/{mlp_name}/{feat_idx}.pt').to_dense()
        split_idx = int(acts.shape[0]*train_amt)

        train_acts, test_acts = acts[:split_idx], acts[split_idx:]
        train_tok_ids, test_tok_ids = self.all_tok_ids[:split_idx], self.all_tok_ids[split_idx:acts.shape[0]]
        print('b')
        synth = tokre.SynthFeat(pattern)
        print('c')
        synth.train(train_acts, train_tok_ids, n_actors=1)
        print('d')
        synth_test_acts = synth.get_acts(test_tok_ids, n_actors=1)
        print('e')
        test_errs = test_acts - synth_test_acts
        print('f')
        r_squared = 1 - test_errs.var()/test_acts.var()
        print('g')
        return r_squared

actors = [Actor.remote() for _ in range(tokre.tot_cpus())]

unprocessed_feats = []
for feat_idx in tqdm(range(100, TOT_FEATS)):
    if f'{feat_idx}.json' in os.listdir(f'tokenset_feats/{MLP_NAME}'):
        continue
    unprocessed_feats.append(feat_idx)

results = ray.get([actors[i%len(actors)].process_feat.remote(feat_idx) for i, feat_idx in enumerate(unprocessed_feats)])

