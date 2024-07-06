from functools import lru_cache
from huggingface_hub import hf_hub_download
import torch
import tiny_model
import pysvelte
from datasets import load_dataset

dataset = load_dataset('noanabeshima/TinyModelTokIds', split='train[:13%]')
all_tok_ids = torch.tensor(dataset['tok_ids'])

@lru_cache
def load_sparse_feat_acts(fname):
    local_path = hf_hub_download(repo_id="noanabeshima/tiny_model_cached_acts", filename=fname)
    csr_kwargs = torch.load(local_path)

    # The matrices may be stored in space-efficient formats that're incompatible with torch's sparse csr tensor.
    # Convert them back before constructing the matrix.
    csr_kwargs['crow_indices'] = csr_kwargs['crow_indices'].int()
    csr_kwargs['col_indices'] = csr_kwargs['col_indices'].int()
    csr_kwargs['values'] = csr_kwargs['values'].float()
    # normalize by the max value of the activations.
    csr_kwargs['values'] = csr_kwargs['values']/csr_kwargs['values'].max()

    assert csr_kwargs['crow_indices'][-1] == csr_kwargs['col_indices'].shape[0]
    assert csr_kwargs['values'].shape == csr_kwargs['col_indices'].shape
    assert csr_kwargs['crow_indices'][-1] == csr_kwargs['values'].shape[0]

    feat_acts = torch.sparse_csr_tensor(**csr_kwargs, size=(csr_kwargs['crow_indices'].shape[0]-1, 128))
    return feat_acts

def get_active_docs(csr_acts):
    crow_indices = csr_acts.crow_indices()
    active_doc_indices = torch.tensor([doc_idx for doc_idx in range(crow_indices.shape[0]-1) if crow_indices[doc_idx+1] != crow_indices[doc_idx]])
    crow_indices_mask = torch.cat((active_doc_indices, active_doc_indices[-1][None]+1))
    active_doc_acts = torch.sparse_csr_tensor(
       crow_indices=crow_indices[crow_indices_mask],
       col_indices=csr_acts.col_indices(),
       values=csr_acts.values(),
       size=(crow_indices_mask.shape[0]-1, 128)
    )

    assert active_doc_acts.shape[0] == active_doc_indices.shape[0]

    return active_doc_acts, active_doc_indices

def load_feat_data(mlp_name, feat_idx, no_zero_docs=True, dense_acts=True):
    global dataset, all_tok_ids
    feat_acts = load_sparse_feat_acts(f'mlp_map_test/{mlp_name}/{feat_idx}.pt')
    tok_ids = all_tok_ids[:feat_acts.shape[0]]
    if no_zero_docs:
        feat_acts, active_doc_indices = get_active_docs(feat_acts)
        tok_ids = torch.tensor(dataset[active_doc_indices]['tok_ids'])
    else:
        tok_ids = torch.tensor(dataset[:feat_acts.shape[0]]['tok_ids'])
    
    toks = tiny_model.toks[tok_ids]

    assert feat_acts.shape == tok_ids.shape, print(feat_acts.shape, tok_ids.shape)
    assert feat_acts.shape == toks.shape

    if dense_acts is True:
        feat_acts = feat_acts.to_dense()

    return feat_acts, tok_ids, toks

def load_feat_splits(mlp_name, feat_idx, train_amt=0.8):
    feat_acts, tok_ids, toks = load_feat_data(mlp_name, feat_idx)
    feat_acts, tok_ids = feat_acts, tok_ids 

    split_idx = int(feat_acts.shape[0]*train_amt)
    
    train_acts, test_acts = feat_acts[:split_idx], feat_acts[split_idx:]
    train_tok_ids, test_tok_ids = tok_ids[:split_idx], tok_ids[split_idx:]

    return train_acts, train_tok_ids, test_acts, test_tok_ids

def see_feat(mlp_name, feat, no_zero_docs=True, **kwargs):
    feat_acts, tok_ids, toks  = load_feat_data(f"{mlp_name}", feat_idx=feat, no_zero_docs=no_zero_docs)

    return pysvelte.WeightedDocs(acts=feat_acts.tolist(), docs=toks.tolist(), **kwargs)

def weighted_docs(acts, doc_ids, **kwargs):
    if isinstance(acts, list):
        acts = torch.tensor(acts)
    
    if isinstance(doc_ids, list):
        docs = torch.tensor(doc_ids)
    
    docs = tiny_model.toks[doc_ids]

    if len(acts.shape) == 1:
        acts = acts[None]
    if len(docs.shape) == 1:
        docs = docs[None]
    
    return pysvelte.WeightedDocs(acts=acts.tolist(), docs=docs.tolist(), **kwargs)