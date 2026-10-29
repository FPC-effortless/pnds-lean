"""Independent functional single-token043 graph. No TAC/scans/traces imported.

Same PyTorch primitive/AD runtime; full IEEE/formal execution bridge OPEN.
"""
import math
import torch
from torch.nn import functional as F

SHAPES={'native':(8,8),'aligned':(2,16,8),'wave':(8,2),'concepts':(6,16),'basis':(88,),
        'lexical':(3,48),'previous_key':(8,),'upper':(48,),'last_logits':(256,)}

def initial(weights,batch):
    return {name:weights['embedding.weight'].new_zeros((batch,*shape)) for name,shape in SHAPES.items()}


def step(weights,s,token,t,mask,mode='native'):
    b=token.shape[0];dtype=weights['embedding.weight'].dtype;device=token.device
    def linear(n,x): return F.linear(x,weights[n+'.weight'],weights.get(n+'.bias'))
    def unit(x): return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    def ordinal(clock):
        values=[]
        for period in (7,13,29,59):
            angle=torch.tensor(clock%period,dtype=dtype,device=device)*(2*math.pi/period)
            values.extend((angle.cos(),angle.sin()))
        return torch.stack(values).expand(b,-1)
    clock=min(t,127) if mode=='cap_all' else t
    tt=torch.tensor(clock,dtype=dtype,device=device)
    position=torch.stack((torch.log1p(tt),torch.sin(tt/16),torch.cos(tt/16),torch.sin(tt/64),torch.cos(tt/64),torch.sin(tt/256),torch.cos(tt/256)))
    embed=F.embedding(token,weights['embedding.weight']);lex=embed.tanh()
    e=(embed+linear('pos',position)).tanh();local=lex
    for lag in range(3): local=local+weights['context_lag_weights'][lag].tanh()*s['lexical'][:,lag]
    key=unit(linear('native_key',local.tanh()));wk=s['previous_key'];value=linear('native_value',e).tanh()
    ab=linear('native_gate',e).sigmoid();retained=.995*ab[:,0,None,None]*s['native']
    native_offer=retained+ab[:,1,None,None]*(value-(retained@wk[:,:,None]).squeeze(-1))[:,:,None]*wk[:,None,:]
    native=torch.where((mask & (t>0))[:,None,None],native_offer,s['native'])
    nread=(native@key[:,:,None]).squeeze(-1)
    ak=unit(torch.einsum('bi,hij->bhj',ordinal(t),weights['rotation']))
    pk=unit(torch.einsum('bi,hij->bhj',ordinal(max(0,t-1)),weights['rotation']))
    ar=(s['aligned']@pk[:,:,:,None]).squeeze(-1).mean(1) if t else e.new_zeros((b,16))
    ag=linear('aligned_gate',e).sigmoid().view(b,2,3);av=linear('aligned_value',e).tanh()
    retained=ag[:,:,0,None,None]*s['aligned']
    residual=(retained@ak[:,:,:,None]).squeeze(-1)
    proposed=retained-ag[:,:,1,None,None]*residual[:,:,:,None]*ak[:,:,None,:]+ag[:,:,2,None,None]*av[:,None,:,None]*ak[:,:,None,:]
    aligned=torch.where(mask[:,None,None,None],proposed,s['aligned'])
    oldwave=s['wave'].flatten(1);ctx=s['basis']/max(1,t)
    fast=linear('fast',torch.cat((e,ctx,oldwave,nread),-1)).tanh()
    slow=linear('slow',torch.cat((fast,ctx,nread),-1)).tanh()
    G=linear('evidence',torch.cat((e,fast,slow,ar,oldwave),-1)).tanh().view(b,6,16)
    sm=(linear('concept_q',G)@linear('concept_k',G).transpose(-1,-2)/4).softmax(-1)
    eye=torch.eye(6,dtype=dtype,device=device);A=.5*(sm+sm.transpose(-1,-2))*(1-eye)
    L=torch.diag_embed(A.sum(-1))-A
    posterior=torch.linalg.solve(1.25*eye+.25*L,G+.25*s['concepts'])
    pool=(posterior*G.mean(-1).softmax(-1)[:,:,None]).sum(1)
    z=F.layer_norm(torch.cat((fast,slow,nread,ar,oldwave,pool,e),-1),(280,),weights['layernorm.0.weight'],weights['layernorm.0.bias'],1e-5)
    unorm=F.layer_norm(z,(280,),weights['layernorm.1.input_norm.weight'],weights['layernorm.1.input_norm.bias'],1e-5)
    u=linear('layernorm.1.input_proj',unorm).tanh()
    retain=.995*linear('layernorm.1.forget',u).sigmoid();offer=linear('layernorm.1.offer',u).tanh()
    gate=linear('layernorm.1.read_gate',torch.cat((u,s['upper']),-1)).sigmoid()
    refined=z+weights['layernorm.1.residual_gain'].sigmoid()*linear('layernorm.1.output',(u+gate*s['upper']).tanh())
    logits=linear('ffn_out',linear('ffn_up',refined)*F.silu(linear('ffn_gate',refined)))+linear('concept_out',pool)
    angle=weights['wave_angle'];rho=.97+.029*linear('wave_decay',e).sigmoid()
    xx=s['wave'][:,:,0]*angle.cos()+s['wave'][:,:,1]*angle.sin()
    yy=-s['wave'][:,:,0]*angle.sin()+s['wave'][:,:,1]*angle.cos()
    wave=rho[:,:,None]*torch.stack((xx,yy),-1)+.025*linear('wave_input',e).tanh().view(b,8,2)
    cg=.05*linear('concept_gate',e).sigmoid().view(b,6,16)
    return logits,{'native':native,'aligned':aligned,'wave':wave,
        'concepts':(1-cg)*s['concepts']+cg*linear('concept_value',e).tanh().view(b,6,16),
        'basis':s['basis']+linear('basis',e).tanh(),'lexical':torch.stack((lex,s['lexical'][:,0],s['lexical'][:,1]),1),
        'previous_key':key,'upper':retain*s['upper']+(1-retain)*offer,'last_logits':logits}
