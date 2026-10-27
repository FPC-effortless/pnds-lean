"""Adversarial tests for the kernel theorem axiom receipt parser."""
from __future__ import annotations
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from audit_language_lean import verify_audit

SOURCE = "#print axioms PNDS.LanguageFoo.ok\n#print axioms PNDS.LanguageFoo.clean\n"
GOOD = (
    "'PNDS.LanguageFoo.ok' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
    "'PNDS.LanguageFoo.clean' does not depend on any axioms\n"
)

class LanguageAxiomAuditTests(unittest.TestCase):
    def test_valid_ordinary_and_zero_axiom_results(self):
        a=verify_audit(SOURCE,GOOD,2)
        self.assertEqual(a["checked_theorems"],2)
        self.assertEqual(a["zero_axiom_theorems"],1)

    def test_long_lean_declaration_multiline_axiom_list(self):
        wrapped=GOOD.replace(
            "[propext, Classical.choice, Quot.sound]",
            "[propext,\n Classical.choice,\n Quot.sound]")
        self.assertEqual(verify_audit(SOURCE,wrapped,2)["checked_theorems"],2)
        injected=wrapped.replace("Quot.sound]", "Quot.sound,\n Dangerous.unproven]")
        with self.assertRaisesRegex(ValueError,"unexpected axioms"):
            verify_audit(SOURCE,injected,2)

    def test_unknown_axiom_not_hidden_by_standard_axiom(self):
        bad=GOOD.replace("Quot.sound]", "Quot.sound, UserDefined.unsafe]")
        with self.assertRaisesRegex(ValueError,"unexpected axioms"):
            verify_audit(SOURCE,bad,2)

    def test_no_missing_or_duplicate_declarations(self):
        with self.assertRaisesRegex(ValueError,"missing"):
            verify_audit(SOURCE,GOOD.splitlines(keepends=True)[0],2)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            verify_audit(SOURCE,GOOD+GOOD.splitlines(keepends=True)[0],2)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            verify_audit(SOURCE+SOURCE.splitlines(keepends=True)[0],GOOD,3)

    def test_no_unexpected_theorem_name(self):
        with self.assertRaisesRegex(ValueError,"extra"):
            verify_audit(SOURCE,GOOD.replace("LanguageFoo.ok","LanguageFoo.unregistered"),2)

    def test_reject_sorry_and_lean_error(self):
        with self.assertRaisesRegex(ValueError,"sorryAx"):
            verify_audit(SOURCE,GOOD+"sorryAx\n",2)
        with self.assertRaisesRegex(ValueError,"error"):
            verify_audit(SOURCE,GOOD+"AuditLanguage.lean:3:2: error: unknown constant\n",2)

    def test_expected_manifest_count_is_binding(self):
        with self.assertRaisesRegex(ValueError,"expected 30"):
            verify_audit(SOURCE,GOOD,30)

if __name__=="__main__":
    unittest.main()
