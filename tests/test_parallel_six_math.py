import json
from pathlib import Path
import random
import unittest

from foundation.v1.parallel_six import parallel_math_reference as ref

ROOT=Path(__file__).resolve().parents[1]
SIX=ROOT/"foundation/v1/parallel_six/active-variants.json"
OLD=ROOT/"foundation/v1/registry.json"
LINKS=ROOT/"foundation/v1/variant-math-links.json"


def error(a,b):
    return max(abs(x-y) for ar,br in zip(a,b) for x,y in zip(ar,br))


class ParallelSixMathTests(unittest.TestCase):
    def test_exact_six_and_archive_fidelity(self):
        x=json.loads(SIX.read_text())
        old=json.loads(OLD.read_text())
        links=json.loads(LINKS.read_text())
        self.assertEqual(len(x["active_models"]),6)
        self.assertEqual(len(old["variants"]),17)
        self.assertEqual(len(links["variants"]),17)
        historical={v["id"]:v for v in old["variants"]}
        self.assertEqual({r["historical_id"] for r in x["historical_crosswalk"]},
                         set(historical))
        for row in x["historical_crosswalk"]:
            self.assertEqual(row["source_commit"],
                             historical[row["historical_id"]]["source"]["commit_sha"])
            self.assertTrue(row["active_families"])
        self.assertFalse(x["governance"]["full_lean_formalization_complete"])
        self.assertFalse(x["governance"]["real_world_superiority_established"])

    def test_ordered_affine_chunk_and_rank_one_step(self):
        rng=random.Random(8129)
        m=ref.zeros()
        schedule=[]
        for t in range(24):
            k=[rng.uniform(-.3,.3) for _ in range(8)]
            v=[rng.uniform(-.25,.25) for _ in range(8)]
            a,b=.87,.26
            if t%7==3:  # strict read-only identity
                schedule.append(ref.schedule(k,v,a,b,write=False))
            else:
                schedule.append(ref.schedule(k,v,a,b))
                m=ref.step_rank_one(m,k,v,a,b)
        self.assertLess(error(ref.apply(ref.zeros(),ref.fold(schedule)),m),1e-11)
        p=ref.compose(ref.fold(schedule[:9]),ref.fold(schedule[9:]))
        self.assertLess(error(ref.apply(ref.zeros(),p),m),1e-11)
        idp=ref.schedule([0.]*8,[0.]*8,.8,.3,write=False)
        self.assertEqual(ref.apply(m,idp),m)

    def test_invalid_capacity_or_gates_rejected(self):
        with self.assertRaises(ValueError):
            ref.schedule([2.]+[0.]*7,[.1]*8,.4,.2)
        with self.assertRaises(ValueError):
            ref.schedule([0.]*8,[0.]*8,1.5,.2)


if __name__ == "__main__":
    unittest.main()
