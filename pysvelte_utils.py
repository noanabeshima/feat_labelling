import pysvelte
from utils import *

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