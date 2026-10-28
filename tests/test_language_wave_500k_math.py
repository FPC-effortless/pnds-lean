import unittest
import numpy as np
from foundation.v1.language_wave_500k_math import (
  CANDIDATE,PERSISTENT_FLOATS,ordered_affine_scan,wave_update,
  concept_equilibrium,concept_energy,softmax)
class Lang500KMath(unittest.TestCase):
    def test_accounting(self):
        self.assertEqual(CANDIDATE,499290);self.assertEqual(PERSISTENT_FLOATS,520)
    def test_wave_bound(self):
        a=np.array([.3,-.4])
        for theta in (.1,1.,3.):
            v=wave_update(a,theta,.91,np.zeros(2))
            self.assertLessEqual(np.linalg.norm(v),np.linalg.norm(a)+1e-12)
    def test_scan(self):
        P=np.broadcast_to(.9*np.eye(2),(5,2,2)).copy()
        Q=np.ones((5,3,2))*.1
        s=ordered_affine_scan(P,Q)
        self.assertAlmostEqual(s[-1,0,0],(1-.9**5),places=9)
    def test_concept_equilibrium(self):
        g=np.random.default_rng(44).normal(size=(6,16))
        prior=np.random.default_rng(45).normal(size=(6,16))
        a=np.ones((6,6))*.1;np.fill_diagonal(a,0)
        c,L,H=concept_equilibrium(g,prior,a)
        self.assertTrue(np.allclose(H@c,g+.25*prior))
        self.assertLessEqual(concept_energy(c,g,prior,L),concept_energy(prior,g,prior,L)+1e-9)
    def test_softmax(self):
        p=softmax(np.array([[1.,3.,-2.],[2.,2.,2.]]))
        self.assertTrue(np.all(p>0));self.assertTrue(np.allclose(p.sum(-1),1))
if __name__=='__main__':unittest.main()
