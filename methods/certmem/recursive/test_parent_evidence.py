from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from parent_evidence import minimum_parent_cover


class ParentCoverTests(unittest.TestCase):
    def test_cover_preserves_the_joint_requirement_and_selects_whole_existing_units(self):
        parent=[{'text':'long_both','sources':['a','b']},
                {'text':'aa','sources':['a','extra']},{'text':'bb','sources':['b']}]
        result=minimum_parent_cover(parent,['a','b'],len)
        self.assertEqual(result['parent_indices'],[1,2])
        self.assertEqual(result['sources'],['a','b','extra'])
        self.assertEqual(result['text'],'aa\n\nbb')
        self.assertIsNone(minimum_parent_cover(parent,['missing'],len))


if __name__=='__main__':
    unittest.main()
