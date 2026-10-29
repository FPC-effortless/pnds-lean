import random,unittest
from foundation.v1.langmem084_write_read import byte_code,read,source_update,overwrite

class WriteReadTests(unittest.TestCase):
    def test_injective_byte_codes(self):
        codes=[byte_code(i) for i in range(256)]
        self.assertEqual(len(set(codes)),256)
        for c in codes:self.assertAlmostEqual(sum(x*x for x in c),1.)
    def test_unit_overwrite_property_and_source_reference(self):
        rng=random.Random(84)
        for _ in range(32):
            M=[[rng.uniform(-1,1) for _ in range(8)] for _ in range(8)]
            k=byte_code(rng.randrange(256));v=byte_code(rng.randrange(256))
            updated=overwrite(M,k,v,.99)
            for x,y in zip(read(updated,k),v):self.assertAlmostEqual(x,y)
            source=source_update(M,k,v,.99,.02,.005)
            for i in range(8):
                for j in range(8):self.assertAlmostEqual(source[i][j],.99*M[i][j]-.99*.02*read(M,k)[i]*k[j]+.005*v[i]*k[j])
if __name__=='__main__':unittest.main()
