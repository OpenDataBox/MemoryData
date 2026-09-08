from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from budgeted_evidence import multiple_choice_budget,storage_cost,probe_options
from evidence_fidelity import accepted


class BudgetTests(unittest.TestCase):
    def test_verifier_does_not_accept_negative_or_ambiguous_labels(self):
        self.assertTrue(accepted('Equivalent.'))
        for value in ('NOT EQUIVALENT','DIFFERENT','UNKNOWN','EQUIVALENT or DIFFERENT','EQUIVALENT\nBut unsupported'):
            self.assertFalse(accepted(value))

    def test_multiple_choice_optimizer_avoids_greedy_expensive_choice(self):
        groups=[[{'id':'a_big','cost':8,'gain':0.9},{'id':'a_small','cost':4,'gain':0.7}],
                [{'id':'b','cost':4,'gain':0.7}]]
        selected,summary=multiple_choice_budget(groups,8,quantum=1)
        self.assertEqual({o['id'] for o in selected},{'a_small','b'})
        self.assertAlmostEqual(summary['gain'],1.4)

    def test_conservative_rounding_cannot_overspend_and_never_selects_two_options_for_one_probe(self):
        groups=[[{'id':'a','cost':5,'gain':0.5},{'id':'b','cost':6,'gain':0.8}],
                [{'id':'c','cost':5,'gain':0.6}]]
        selected,summary=multiple_choice_budget(groups,10,quantum=4)
        self.assertEqual([o['id'] for o in selected],['b'])
        self.assertLessEqual(summary['actual_tokens'],10)

    def test_storage_counts_distinct_index_and_payload_and_rejects_audit_input(self):
        self.assertEqual(storage_cost({'text':'abcd','index_text':'xy'},len),6)
        self.assertEqual(storage_cost({'text':'abcd','index_text':'abcd'},len),4)
        with self.assertRaises(ValueError):
            probe_options([], [{'id':'audit','split':'probe_audit'}],len,'joint')


if __name__=='__main__':
    unittest.main()
