import importlib.util,random,sys,unittest
from pathlib import Path
path=Path(__file__).resolve().parents[1]/"foundation/v1/langmem080_source_event.py"
spec=importlib.util.spec_from_file_location("langmem080_source_event",path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class HistoricalNativeWrite(unittest.TestCase):
    def test_frozen_source_clock_pairs(self):
        self.assertEqual(len(m.PAIRS),27)
        self.assertTrue(all(a+d<320 and d>0 for a,d in m.PAIRS))
        self.assertNotIn((192,192),m.PAIRS)
    def test_additive_specific_original_event_exact_real(self):
        rng=random.Random(8004)
        for _ in range(18):
            q=[[rng.uniform(-.1,.1) for _ in range(m.DIM)] for _ in range(m.DIM)]
            n=[[rng.uniform(-1,1) for _ in range(m.DIM)] for _ in range(m.DIM)]
            p=[[rng.uniform(-.2,.2) for _ in range(m.DIM)] for _ in range(m.DIM)]
            c=[[rng.uniform(-.2,.2) for _ in range(m.DIM)] for _ in range(m.DIM)]
            whole=m.step(n,p,q)
            without=m.step(m.subtract(n,c),p,q)
            original_tag=m.tag_step(c,p)
            actual=m.subtract(whole,without)
            self.assertLess(m.norm(m.subtract(actual,original_tag)),1e-12)
            self.assertEqual(m.step(m.subtract(n,m.zero()),p,q),whole)
    def test_event_write_superposition_across_observed_stream(self):
        rng=random.Random(8005)
        n=m.zero()
        tags=[]
        for t in range(18):
            p=[[.8*(i==j)+.01*rng.random() for j in range(8)] for i in range(8)]
            q=[[rng.uniform(-.01,.01) for _ in range(8)] for _ in range(8)]
            n=m.step(n,p,q)
            tags=[m.tag_step(c,p) for c in tags]+[q]
            summed=m.zero()
            for tagged in tags:summed=m.add(summed,tagged)
            self.assertLess(m.norm(m.subtract(summed,n)),1e-12)
    def test_malformed_shape(self):
        with self.assertRaises(ValueError):m.step(m.zero(),m.zero(),[[1.]])
