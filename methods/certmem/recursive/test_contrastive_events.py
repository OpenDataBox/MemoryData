from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from contrastive_events import proposal_pairs,choose_pairs,construct,original_context


class ContrastiveEventTests(unittest.TestCase):
    def test_graph_requires_different_sessions_and_nonoverlapping_sources(self):
        import numpy as np
        units=[{'session':1,'sources':['a']},{'session':1,'sources':['b']},
               {'session':2,'sources':['a']},{'session':3,'sources':['c']}]
        edges=proposal_pairs(units,np.ones((4,2)),limit=8,max_degree=1)
        indices=[i for e in edges for i in e['parent_indices']]
        self.assertEqual(len(indices),len(set(indices)))
        for edge in edges:
            a,b=[units[i] for i in edge['parent_indices']]
            self.assertNotEqual(a['session'],b['session'])
            self.assertFalse(set(a['sources'])&set(b['sources']))

    def test_controls_choose_identical_pairs_under_common_cost_and_cannot_duplicate_sources(self):
        def candidate(name,sources,cost,similarity,accepted=True):
            return {'id':name,'unit':{'text':'payload','index_text':'key','sources':sources,'session':1},
                    'common_cost':cost,'accepted':accepted,'edge':{'similarity':similarity}}
        values=[candidate('x',['a','b'],10,.9),candidate('y',['b','c'],8,.8),
                candidate('z',['d'],10,.6),candidate('bad',['e'],1,1,False)]
        selected,details=choose_pairs(values,20)
        self.assertEqual(details['selected_ids'],['y','z'])
        raw,r=construct([],values,len,'literal',20)
        keyed,k=construct([],values,len,'contrastive',20)
        self.assertEqual(r['selected_ids'],k['selected_ids'])
        self.assertEqual([u['text'] for u in raw],[u['text'] for u in keyed])
        self.assertTrue(all('index_text' not in u for u in raw))
        self.assertTrue(all('index_text' in u for u in keyed))
        self.assertEqual(choose_pairs(values,0)[0],[])

    def test_key_authority_is_original_source_with_session_attribution(self):
        sessions=[{'date':'day one','turns':[{'id':'a','text':'Original A'}]},
                  {'date':'day two','turns':[{'id':'b','text':'Original B'}]}]
        self.assertEqual(original_context(sessions,['b','a']),
                         '(Session date: day one) Original A\n\n(Session date: day two) Original B')
        with self.assertRaises(ValueError):
            original_context(sessions,['missing'])


if __name__=='__main__':
    unittest.main()
