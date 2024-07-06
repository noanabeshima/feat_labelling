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

ray.init(ignore_reinit_error=True)

def test_pattern(mlp_name, feat_idx, pattern):
    train_acts, train_tok_ids, test_acts, test_tok_ids = load_feat_splits(MLP_NAME, feat_idx, train_amt=0.8)
    synth = tokre.SynthFeat(pattern)
    synth.train(train_acts, train_tok_ids)
    synth_test_acts = synth.get_acts(test_tok_ids)
    test_errs = test_acts - synth_test_acts
    r_squared = 1 - test_errs.var()/test_acts.var()
    return r_squared

def process_feat(feat_idx):
    feat_acts = load_sparse_feat_acts(f'{MLP_DIR}/{MLP_NAME}/{feat_idx}.pt').to_sparse_coo()
    indices = feat_acts.indices()[:,:600]
    active_tok_ids = all_tok_ids[indices[0], indices[1]]

    uniq, counts = torch.unique(active_tok_ids, return_counts=True)
    
    num_unique = uniq.shape[0]
    topk_counts, _topk_indices = counts.topk(k=min(num_unique, 5))
    topk_ids = uniq[_topk_indices]

    pattern = "(" + "|".join([tokre.escape(tok) for tok in tiny_model.raw_toks[topk_ids]]) + ")"
    try:
        
        pos_r_squared = test_pattern(MLP_NAME, feat_idx, pattern + '[pos]')

        data = {
            'pattern': pattern,
            'pos_r_squared': pos_r_squared.item()
        }

        if pos_r_squared > 0.98:
            r_squared = test_pattern(MLP_NAME, feat_idx, pattern)
            data['r_squared'] = r_squared.item()
        
        
        output_dir = os.path.join('tokenset_feats', MLP_NAME)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'{feat_idx}.json')
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(pattern)
        raise e

results = []
for feat_idx in tqdm(range(100, TOT_FEATS)):
    if f'{feat_idx}.json' in os.listdir(f'tokenset_feats/{MLP_NAME}'):
        continue
    results.append(process_feat.remote(feat_idx))

ray.get(results)
    

