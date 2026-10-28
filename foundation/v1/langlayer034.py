"""Independent serial PyTorch operator for one LANGLAYER034 residual block.

No imports from TAC, no scan helpers, no source trace inputs. This reference
covers ONLY an extra residual layer given its source-legal 280-wide input;
the inherited complete CONTEXT026 graph is formulated separately under #62.
PyTorch autograd is shared; no independent AD or IEEE proof is implied.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def serial_residual_layer(weights: dict[str, torch.Tensor], z: torch.Tensor):
    if z.ndim != 3 or z.shape[-1] != 280 or z.shape[1] == 0:
        raise ValueError("source features must have shape [B,T,280]")
    b, length, _ = z.shape

    def affine(prefix, x):
        return F.linear(x, weights[prefix+'.weight'], weights.get(prefix+'.bias'))

    s = weights['input_proj.weight'].shape[0]
    h = z.new_zeros((b, s))
    c = z.new_zeros((b, 6, 16))
    outputs = []
    states = []
    prev_states = []
    concepts = []
    laplacians = []
    eye = torch.eye(6, device=z.device, dtype=z.dtype)
    for t in range(length):
        u = affine('input_proj', F.layer_norm(
            z[:, t], (280,), weights['input_norm.weight'],
            weights['input_norm.bias'], eps=1e-5)).tanh()
        old_h = h
        retain = .995 * torch.sigmoid(affine('forget', u))
        offered = torch.tanh(affine('offer', u))
        h = retain * h + (1-retain) * offered
        read_gate = torch.sigmoid(affine('read_gate', torch.cat((u, old_h), -1)))
        old_c = c
        rate = .05 * torch.sigmoid(affine('concept_gate', u)).reshape(b, 6, 16)
        concept_offer = affine('concept_offer', u).tanh().reshape(b, 6, 16)
        c = (1-rate) * c + rate * concept_offer
        evidence = concept_offer + old_c
        query = affine('concept_q', evidence)
        key = affine('concept_k', evidence)
        soft = (query @ key.transpose(-1, -2) / 4).softmax(-1)
        adjacency = .5 * (soft + soft.transpose(-1, -2)) * (1-eye)
        laplacian = torch.diag_embed(adjacency.sum(-1)) - adjacency
        posterior = torch.linalg.solve(
            1.25 * eye + .25 * laplacian, evidence + .25 * old_c)
        pool = (posterior * evidence.mean(-1).softmax(-1).unsqueeze(-1)).sum(-2)
        delta = (u + read_gate * old_h +
                 affine('concept_to_state', pool)).tanh()
        outputs.append(z[:, t] + weights['residual_gain'].sigmoid() *
                       affine('output', delta))
        prev_states.append(old_h)
        states.append(h)
        concepts.append(posterior)
        laplacians.append(laplacian)
    return torch.stack(outputs, 1), {
        'state': torch.stack(states, 1),
        'prior_state': torch.stack(prev_states, 1),
        'concept_posterior': torch.stack(concepts, 1),
        'concept_laplacian': torch.stack(laplacians, 1),
    }
