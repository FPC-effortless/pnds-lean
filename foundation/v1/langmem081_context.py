"""LANGMEM081 label-blind past-context reference only.

Pure integer source prefix and wrong-history selection, not original
trained PyTorch keys. Content equality and side-table memory are NOT
TAC's fixed 4160-byte persistent state. Future byte is omitted entirely.
"""
import hashlib
LENGTHS=(2,3,4)
AGES=(32,128,256)
def earlier_candidates(observed_prefix,age_cap):
    if type(observed_prefix) is not bytes or len(observed_prefix)<129 or age_cap not in AGES:
        raise ValueError("causal original prefix/horizon")
    t=len(observed_prefix)-1
    return tuple(range(max(4,t-age_cap),t))

def matching_indices(observed_prefix,L,H):
    if L not in LENGTHS:raise ValueError("registered observed context length only")
    indices=earlier_candidates(observed_prefix,H)
    t=len(observed_prefix)-1
    query=observed_prefix[t-L+1:t+1]
    return tuple(a for a in indices if observed_prefix[a-L:a]==query)

def most_recent_past_match(observed_prefix,L,H):
    matches=matching_indices(observed_prefix,L,H)
    return matches[-1] if matches else None

def deterministic_wrong_or_uniform_index(owner,observed_prefix,L,H):
    if type(owner) is not str or not owner:
        raise ValueError("original source owner identity required")
    indices=earlier_candidates(observed_prefix,H)
    t=len(observed_prefix)-1
    seed=hashlib.sha256(f"LANGMEM081|{owner}|{t}|{L}|{H}".encode()).digest()
    return indices[int.from_bytes(seed[:8],"big")%len(indices)]
