import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('c18_scoped_axioms',Path(__file__).resolve().parents[1]/'scripts/check_collect018_axioms.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class Scoped(unittest.TestCase):
    def test_exact_reviewed_sets_and_reject_false_receipts(self):
        good='\n'.join("'"+n+"' depends on axioms: [propext, Classical.choice, Quot.sound]" for n in audit.EXPECTED)
        self.assertEqual(audit.check(good)['status'],'PASS_EXACT_SCOPED_AXIOM_SETS')
        for bad in ('',good.splitlines()[0],good+'\nsorryAx',good+'\nerror: bad',good+'\n'+good.splitlines()[0],
                    good.replace('Quot.sound','customAssumption',1),good.replace('propext, ','',1),
                    good.replace('Quot.sound]','Quot.sound, Quot.sound]',1),good.replace('depends on axioms: [propext, Classical.choice, Quot.sound]','does not depend on any axioms',1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):audit.check(bad)
