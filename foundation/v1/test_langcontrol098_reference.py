"""Constructed equation checks; no training or capability evidence."""
import unittest
import numpy as np
from langcontrol098_reference import coefficient_array, controlled_cross_entropy


def sample():
    # Unequal domain cardinalities AND episode segment lengths expose averaging bugs.
    valid = np.array([[1,1,1,1,0], [1,1,1,1,1], [1,1,1,0,0]], dtype=bool)
    future = np.array([[0,0,1,1,0], [0,1,1,1,1], [0,0,1,0,0]], dtype=bool)
    domains = ['lexical', 'lexical', 'arc']
    logits = np.random.default_rng(98001).normal(size=(3,5,4))
    targets = np.array([[0,1,2,3,0], [2,3,1,0,2], [1,0,2,0,0]])
    return logits, targets, future, valid, domains


class ControlledReferenceTests(unittest.TestCase):
    def test_y_coefficients_identical_and_explicit(self):
        _, _, future, valid, domains = sample()
        zero = coefficient_array(future, valid, domains, prefix_weight=0)
        one = coefficient_array(future, valid, domains, prefix_weight=1)
        np.testing.assert_array_equal(zero[future], one[future])
        np.testing.assert_array_equal(zero[0, future[0]], [.125, .125])
        np.testing.assert_array_equal(zero[1, future[1]], [.0625]*4)
        self.assertEqual(zero[2, 2], .5)

    def test_prefix_and_padding_coefficients(self):
        _, _, future, valid, domains = sample()
        zero = coefficient_array(future, valid, domains, prefix_weight=0)
        one = coefficient_array(future, valid, domains, prefix_weight=1)
        np.testing.assert_array_equal(zero[~future], np.zeros((~future).sum()))
        np.testing.assert_array_equal(one[~valid], np.zeros((~valid).sum()))
        self.assertEqual(one[0,0], .125)
        self.assertEqual(one[1,0], .25)
        self.assertEqual(one[2,0], .25)
        self.assertEqual(zero.sum(), 1.)
        self.assertEqual(one.sum(), 2.)

    def test_independent_scalar_episode_mean(self):
        z, y, future, valid, domains = sample()
        for lam in (0,1):
            totals = {}
            for d in ('lexical','arc'):
                episodes = []
                for i in range(3):
                    if domains[i] != d: continue
                    ce = [float(np.log(np.exp(z[i,t]-max(z[i,t])).sum())
                                - (z[i,t,y[i,t]]-max(z[i,t]))) for t in range(5)]
                    yy=[ce[t] for t in range(5) if future[i,t]]
                    pp=[ce[t] for t in range(5) if valid[i,t] and not future[i,t]]
                    episodes.append(sum(yy)/len(yy)+lam*sum(pp)/len(pp))
                totals[d]=sum(episodes)/len(episodes)
            actual,_ = controlled_cross_entropy(z,y,future,valid,domains,prefix_weight=lam)
            self.assertAlmostEqual(actual,.5*(totals['lexical']+totals['arc']),places=14)

    def test_logit_gradient_finite_difference(self):
        z,y,future,valid,domains=sample()
        for lam in (0,1):
            _,grad=controlled_cross_entropy(z,y,future,valid,domains,prefix_weight=lam)
            for index in np.ndindex(z.shape):
                plus,minus=z.copy(),z.copy();plus[index]+=1e-6;minus[index]-=1e-6
                a=controlled_cross_entropy(plus,y,future,valid,domains,prefix_weight=lam)[0]
                b=controlled_cross_entropy(minus,y,future,valid,domains,prefix_weight=lam)[0]
                self.assertAlmostEqual(grad[index],(a-b)/2e-6,delta=5e-10)

    def test_prefix_label_poison_only_affects_prefix_arm(self):
        z,y,future,valid,domains=sample();changed=y.copy()
        changed[valid&~future]=(changed[valid&~future]+1)%4
        a,ga=controlled_cross_entropy(z,y,future,valid,domains,prefix_weight=0)
        b,gb=controlled_cross_entropy(z,changed,future,valid,domains,prefix_weight=0)
        self.assertEqual(a,b);np.testing.assert_array_equal(ga,gb)
        a,_=controlled_cross_entropy(z,y,future,valid,domains,prefix_weight=1)
        b,_=controlled_cross_entropy(z,changed,future,valid,domains,prefix_weight=1)
        self.assertNotEqual(a,b)

    def test_padding_label_poison_irrelevant_in_both_arms(self):
        z,y,future,valid,domains=sample();changed=y.copy();changed[~valid]=3
        for lam in (0,1):
            a,ga=controlled_cross_entropy(z,y,future,valid,domains,prefix_weight=lam)
            b,gb=controlled_cross_entropy(z,changed,future,valid,domains,prefix_weight=lam)
            self.assertEqual(a,b);np.testing.assert_array_equal(ga,gb)

    def test_bad_coefficients_rejected(self):
        _,_,f,v,d=sample()
        for lam in (-1,.5,2,True,np.nan):
            with self.subTest(lam=lam),self.assertRaises(ValueError):
                coefficient_array(f,v,d,prefix_weight=lam)

    def test_missing_domain_and_empty_segments_rejected(self):
        _,_,f,v,d=sample()
        cases=[(f,v,['lexical']*3),(v,v,d),(np.zeros_like(f),v,d)]
        invalid=f.copy();invalid[0,4]=True;cases.append((invalid,v,d))
        for future,valid,domains in cases:
            with self.assertRaises(ValueError):coefficient_array(future,valid,domains,prefix_weight=0)

    def test_extreme_finite_logits_fail_closed(self):
        z,y,f,v,d=sample();z[:]=1e308;z[...,0]=-1e308
        for lam in (0,1):
            with self.assertRaises(ValueError):
                controlled_cross_entropy(z,y,f,v,d,prefix_weight=lam)

    def test_finite_terms_with_overflowing_total_fail_closed(self):
        _,_,f,v,d=sample();z=np.zeros((3,5,4));z[...,0]=1e308
        y=np.ones((3,5),dtype=int)
        with self.assertRaises(ValueError):
            controlled_cross_entropy(z,y,f,v,d,prefix_weight=1)

    def test_invalid_logit_or_target_rejected(self):
        z,y,f,v,d=sample()
        for badz,bady in ((z*np.nan,y),(z,y.astype(float)),(z,y+4)):
            with self.assertRaises(ValueError):
                controlled_cross_entropy(badz,bady,f,v,d,prefix_weight=0)


if __name__=='__main__':unittest.main()
