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
CORES_PER_ACTOR = 1

import numpy as np
from utils import all_tok_ids
import os
from tqdm import tqdm
import ray
import time

ray.init(ignore_reinit_error=True)#, object_store_memory=60)



@ray.remote(num_cpus=CORES_PER_ACTOR)
class Actor:
    def __init__(self, all_tok_ids):
        self.all_tok_ids = all_tok_ids
        tokre.setup(tokenizer=tiny_model.tokenizer, workspace='workspace')

    def process_feat(self, feat_idx):
        ray.logger.info(f'starting to process {feat_idx}')
        feat_acts = load_sparse_feat_acts(f'{MLP_DIR}/{MLP_NAME}/{feat_idx}.pt').to_sparse_coo()
        # ray.logger.info('loaded feat acts')
        indices = feat_acts.indices()[:,:600]
        # ray.logger.info('sliced indices')
        active_tok_ids = self.all_tok_ids[indices[0], indices[1]]

        uniq, counts = torch.unique(active_tok_ids, return_counts=True)
        
        num_unique = uniq.shape[0]
        topk_counts, _topk_indices = counts.topk(k=min(num_unique, 5))
        topk_ids = uniq[_topk_indices]

        pattern = "(" + "|".join([tokre.escape(tok) for tok in tiny_model.raw_toks[topk_ids]]) + ")"
        try:
            # ray.logger.info(f'starting to test pattern {feat_idx}')
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
            # ray.logger.info(f'saving {feat_idx}.json')
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            ray.logger.info(f'Saved {feat_idx}.json')
        except Exception as e:
            ray.logger.info(pattern)
            ray.logger.info(f"An exception occurred: {type(e).__name__}")
            ray.logger.info(f"Exception message: {str(e)}")
            ray.logger.info("Traceback:")
            import traceback
            traceback.print_exc()
            raise e
        
    

    def test_pattern(self, mlp_name, feat_idx, pattern, train_amt=0.8):
        ray.logger.info(f'loading sparse feat acts in test_pattern')
        acts = load_sparse_feat_acts(f'mlp_map_test/{mlp_name}/{feat_idx}.pt').to_dense()
        split_idx = int(acts.shape[0]*train_amt)
        # train_acts, test_acts = slice_csr(acts, 0, split_idx), slice_csr(acts, split_idx, None).to_dense()
        train_acts, test_acts = acts[:split_idx], acts[split_idx:]
        train_tok_ids, test_tok_ids = self.all_tok_ids[:split_idx], self.all_tok_ids[split_idx:acts.shape[0]]
        synth = tokre.SynthFeat(pattern, disable_parallel=True, disable_tqdm=True)
        synth.train(train_acts, train_tok_ids)#, n_actors=CORES_PER_ACTOR)
        synth_test_acts = synth.get_acts(test_tok_ids)#, n_actors=CORES_PER_ACTOR)
        test_errs = test_acts - synth_test_acts
        r_squared = 1 - test_errs.var()/test_acts.var()
        return r_squared


dataset = load_dataset('noanabeshima/TinyModelTokIds', split='train[:13%]')
all_tok_ids = torch.tensor(dataset['tok_ids'])
actors = [Actor.remote(all_tok_ids) for _ in range(tokre.tot_cpus())]#tokre.tot_cpus()//CORES_PER_ACTOR)]

unprocessed_feats = []
for feat_idx in tqdm(range(0, TOT_FEATS), desc='loading unprocessed_feats'):
    if f'{feat_idx}.json' in os.listdir(f'tokenset_feats/{MLP_NAME}'):
        continue
    unprocessed_feats.append(feat_idx)


BATCH_SIZE = 1000

for batch_idx in range(0, len(unprocessed_feats), BATCH_SIZE):
    results = ray.get([actors[i%len(actors)].process_feat.remote(feat_idx) for i, feat_idx in enumerate(unprocessed_feats[batch_idx:batch_idx+BATCH_SIZE])])