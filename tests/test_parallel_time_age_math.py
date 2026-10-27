import math
import unittest
from foundation.v1.parallel_time_age_math import ClockState, advance


class ParallelTimeAgeMathTests(unittest.TestCase):
    def test_known_timeline_with_nonwrites(self):
        s=ClockState()
        got=[]
        for dt, write in zip((1,2,3,4,5),(True,False,False,True,False)):
            f,s=advance(s,dt,write)
            got.append(f)
        for f, age in zip(got,(1,2,5,9,5)):
            self.assertAlmostEqual(math.expm1(f[5]),age)
        self.assertEqual([f[6] for f in got],[0,1,1,1,1])
        self.assertEqual(s.last_write,10)
        self.assertEqual(s.elapsed,15)

    def test_same_physical_time_distinct_sequence_positions(self):
        f0,s=advance(ClockState(),0.0,False)
        f1,s=advance(s,0.0,False)
        self.assertNotEqual(f0[0],f1[0])
        self.assertEqual(s.elapsed,0.0)

    def test_resume_equals_single_stream(self):
        events=[(1.0,True),(2.,False),(0.,False),(7.,True),(20.,False)]
        def fold(parts, state=ClockState()):
            features=[]
            for dt,w in parts:
                f,state=advance(state,dt,w)
                features.append(f)
            return features,state
        allfeatures,allstate=fold(events)
        a,mid=fold(events[:2])
        b,final=fold(events[2:],mid)
        self.assertEqual(allfeatures,a+b)
        self.assertEqual(allstate,final)

    def test_reject_negative_missing_and_nonfinite(self):
        for dt in (-1.0,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                advance(ClockState(),dt,True)
        with self.assertRaises(ValueError):
            advance(ClockState(),1.0,1)


if __name__=="__main__":
    unittest.main()
