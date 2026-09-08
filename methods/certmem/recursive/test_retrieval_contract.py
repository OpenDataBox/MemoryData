from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from retrieval_contract import compare_contract


class ContractTests(unittest.TestCase):
    def test_more_repairs_cannot_compensate_for_a_preserved_source_behavior_regression(self):
        before={'a':{'preserved':True},'b':{'preserved':False},'c':{'preserved':False}}
        after={'a':{'preserved':False},'b':{'preserved':True},'c':{'preserved':True}}
        result=compare_contract(before,after)
        self.assertFalse(result['feasible'])
        self.assertEqual(result['regressions'],['a'])
        after['a']['preserved']=True
        self.assertTrue(compare_contract(before,after)['feasible'])
        self.assertFalse(compare_contract(before,before)['feasible'])
        with self.assertRaises(ValueError):
            compare_contract(before,{'a':{'preserved':True}})


if __name__=='__main__':
    unittest.main()
