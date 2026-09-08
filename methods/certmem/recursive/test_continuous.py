import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from continuous import calendar_anchor, construct, dialogue_blocks
from refine import session_data
from test_refine import samples, FakeRuntime


class ContinuingProtocolTests(unittest.TestCase):
    def test_calendar_arithmetic_crosses_boundaries_and_preserves_relative_quote(self):
        anchored=calendar_anchor('I left yesterday and came home last night.','9:00 am on 1 March, 2024')
        self.assertIn('29 February 2024',anchored)
        self.assertIn('original relative expression: yesterday',anchored)
        self.assertIn('original relative expression: last night',anchored)
        self.assertIn('December 2023',calendar_anchor('last month','1 January, 2024'))
        self.assertIn('approximately 2023',calendar_anchor('a year ago','1 March, 2024'))

    def test_ambiguous_time_is_not_invented(self):
        self.assertEqual(calendar_anchor('recently in childhood','1 March, 2024'),'recently in childhood')
        self.assertEqual(calendar_anchor('yesterday','unknown date'),'yesterday')
        self.assertIn('12 June 2023 to 18 June 2023',calendar_anchor('last week','19 June, 2023'))

    def test_nonoverlap_dialogue_blocks_keep_every_turn_exactly_once(self):
        sessions=session_data(samples()[0])
        blocks=dialogue_blocks(sessions,size=2)
        self.assertEqual([sid for unit in blocks for sid in unit['sources']],['D1:1','D1:2','D1:3'])

    def test_construction_cannot_see_gold_and_does_not_mutate_parent(self):
        sample=samples()[0]
        sessions=session_data(sample)
        rt=FakeRuntime()
        parent=dialogue_blocks(sessions,size=2)
        old=copy.deepcopy(parent)
        recipe={'operations':[{'op':'extract','prompt':'Extract exact source-linked facts.'},{'op':'anchor_time'}]}
        result=construct(rt,sessions,parent,recipe)
        self.assertEqual(parent,old)
        self.assertNotIn('GOLD_SENTINEL','\n'.join(rt.prompts))
        self.assertNotIn('Question 0','\n'.join(rt.prompts))
        self.assertTrue(all(set(u['sources']) <= {'D1:1','D1:2','D1:3'} for u in result))
        self.assertEqual(result,construct(rt,sessions,result,{'operations':[{'op':'anchor_time'}]}))


if __name__=='__main__':
    unittest.main()
