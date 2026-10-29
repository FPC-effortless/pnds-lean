import random,unittest
from foundation.v1.langmem082_compressed import read,rank_one,permute_rows,norm_sq,ring_available,capacity,norm_matched_wrong_event

class CompressedReadTests(unittest.TestCase):
    def test_original_shape_operator_properties(self):
        rng=random.Random(82)
        for _ in range(32):
            a=[[rng.uniform(-1,1) for _ in range(8)] for _ in range(8)]
            b=[[rng.uniform(-1,1) for _ in range(8)] for _ in range(8)]
            q=[rng.uniform(-1,1) for _ in range(8)]
            total=[[a[i][j]+b[i][j] for j in range(8)] for i in range(8)]
            for x,y,z in zip(read(a,q),read(b,q),read(total,q)):self.assertAlmostEqual(x+y,z)
            self.assertAlmostEqual(norm_sq(a),norm_sq(permute_rows(a)))
            v=[rng.uniform(-1,1) for _ in range(8)];w=[rng.uniform(-1,1) for _ in range(8)]
            event=rank_one(v,q);wrong=norm_matched_wrong_event(event,v,w)
            self.assertAlmostEqual(norm_sq(event),norm_sq(wrong))
    def test_ring_bounds_and_capacity(self):
        self.assertEqual(capacity(256),240);self.assertEqual(capacity(4160),4144)
        self.assertTrue(ring_available(500,240,260));self.assertFalse(ring_available(500,240,259))
        self.assertFalse(ring_available(500,240,500))
if __name__=='__main__':unittest.main()
