import unittest
import numpy as np
from langtrain099_reference import forward,controlled_loss

def parameters(scaled=False):
    rng=np.random.default_rng(98001)
    p={'embedding.weight':rng.normal(0,.3,(256,32)),
       'context_lag':rng.normal(0,.1,(3,32)),
       'raw_context_decay':np.linspace(-8,-2,48),
       'bank_min':np.array([.97,.995]),'bank_max':np.array([.995,.99995]),
       'norm.weight':np.ones(162),'norm.bias':np.zeros(162)}
    for name,ni,no in [('context_input',32,48),('address_value',112,66),('compare',130,16),
                       ('ffn_up',162,138),('ffn_gate',162,138),('ffn_out',138,256)]:
        p[name+'.weight']=rng.normal(0,.1,(no,ni));p[name+'.bias']=rng.normal(0,.03,no)
    if scaled:
        d,c,k,v,w=256,376,24,32,4335;z=2*d+c+3*v+2
        p={'embedding.weight':rng.normal(0,.03,(256,d)), 'context_lag':rng.normal(0,.1,(3,d)),
           'raw_context_decay':np.linspace(-8,-2,c),'bank_min':np.array([.97,.995]),'bank_max':np.array([.995,.99995]),
           'norm.weight':np.ones(z),'norm.bias':np.zeros(z)}
        for name,ni,no in [('context_input',d,c),('address_value',2*d+c,4*k+2*v+2),('compare',d+c+3*v+2,v),
                          ('ffn_up',z,w),('ffn_gate',z,w),('ffn_out',w,256)]:
            p[name+'.weight']=rng.normal(0,.01,(no,ni));p[name+'.bias']=rng.normal(0,.003,no)
    return p

class ReferenceTests(unittest.TestCase):
    def test_exact10m_typed_full_graph(self):
        p=parameters(scaled=True)
        self.assertEqual(sum(v.size for k,v in p.items() if k not in ('bank_min','bank_max')),10000000)
        y,s=forward(p,np.array([[1,2,3]]));self.assertEqual(y.shape,(1,3,256));self.assertTrue(np.isfinite(y).all())
        self.assertEqual(sum(v.size for v in s.values()),2776)
        self.assertTrue((np.abs(s['memory'])<=s['mass'][:,:,None,:]+1e-12).all())

    def test_full_typed_graph_and_state_cone(self):
        p=parameters();x=np.arange(258).reshape(2,129)%256;y,s=forward(p,x)
        self.assertEqual(y.shape,(2,129,256));self.assertTrue(np.isfinite(y).all())
        self.assertTrue((s['mass']>=0).all());self.assertTrue((s['mass'].sum(-1)<=1+1e-12).all())
        self.assertTrue((np.abs(s['memory'])<=s['mass'][:,:,None,:]+1e-12).all())
        self.assertLessEqual(np.abs(s['context']).max(),1)
    def test_arbitrary_chronological_chunk(self):
        p=parameters();x=np.arange(42).reshape(2,21)
        y,s=forward(p,x);a,sa=forward(p,x[:,:7]);b,sb=forward(p,x[:,7:],sa,clock=7)
        np.testing.assert_allclose(y,np.concatenate((a,b),1),atol=1e-12,rtol=1e-12)
        for n in s:np.testing.assert_allclose(s[n],sb[n],atol=1e-12,rtol=1e-12)
    def test_causal_no_future_dependence(self):
        p=parameters();x=np.arange(15)[None];q=x.copy();q[:,8:]=255
        a,_=forward(p,x);b,_=forward(p,q);np.testing.assert_array_equal(a[:,:8],b[:,:8])
    def test_nonwrite_preserves_bank_not_observations(self):
        p=parameters();x=np.arange(7)[None];_,s=forward(p,x)
        _,t=forward(p,x,s,clock=7,write_mask=np.zeros_like(x,dtype=bool))
        for n in ('mass','memory'):np.testing.assert_array_equal(s[n],t[n])
        self.assertFalse(np.array_equal(s['context'],t['context']))
    def test_future_coefficients_independent_of_prefix_coefficient(self):
        rng=np.random.default_rng(98);z=rng.normal(size=(4,9,256));y=np.arange(36).reshape(4,9)
        f=np.zeros_like(y,dtype=bool);f[:2,-2:]=True;f[2:,-5:]=True;valid=np.ones_like(f)
        domains=['code','code','arc','arc'];eps=1e-5;grad=[]
        for weight in (0.,1.,2.):
            plus=z.copy();minus=z.copy();plus[0,8,0]+=eps;minus[0,8,0]-=eps
            grad.append((controlled_loss(plus,y,f,valid,domains,prefix_weight=weight)-controlled_loss(minus,y,f,valid,domains,prefix_weight=weight))/(2*eps))
        np.testing.assert_allclose(grad,grad[0],atol=2e-10,rtol=1e-5)

if __name__=='__main__':unittest.main()
