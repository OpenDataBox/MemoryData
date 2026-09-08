import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from compile_research_data import literal_probes,audit_manifest,split_probe_pool,cards_from_generations
from refine import session_data,make_manifest
from test_refine import samples


class ResearchDataTests(unittest.TestCase):
    def test_multiline_generation_is_grounded_by_source_spans_without_fabricated_ids(self):
        dump={'records':[{'conv_id':'c','session':1,'turn_ids':['D1:1','D1:2'],
                          'text':'Q: Where did Alice go?\nA: Paris\n\nQ: What was closed?\nA: museum\n'}]}
        cards=cards_from_generations(dump)['c']
        self.assertEqual(len(cards),2)
        probes=literal_probes(session_data(samples()[0]),cards)
        self.assertEqual(probes[0]['source_ids'],['D1:1'])
        self.assertEqual(probes[0]['candidate_context_ids'],['D1:1','D1:2'])

    def test_only_literal_supported_answers_are_admitted(self):
        sessions=session_data(samples()[0])
        cards=[{'text':'Q: Where did Alice visit? A: Paris','sources':['D1:1'],'session':1},
               {'text':'Q: Where did Alice visit? A: London','sources':['D1:1'],'session':1},
               {'text':'Q: Where did Alice visit? A: Paris','sources':['D999:1'],'session':1}]
        records=literal_probes(sessions,cards)
        self.assertEqual(len(records),1)
        span=records[0]['answer_spans'][0]
        self.assertEqual(sessions[0]['turns'][0]['body'][span['start']:span['end']],'Paris')

    def test_audit_excludes_old_queries_and_is_independent_of_qa_text(self):
        data=samples()
        old=make_manifest(data)
        first=audit_manifest(data,old,n=15)
        mutated=copy.deepcopy(data)
        for sample in mutated:
            for qa in sample['qa']:
                qa['question']='CHANGED QUESTION'
                qa['answer']='CHANGED GOLD'
                qa['evidence']=['D999:1']
        self.assertEqual(first['records'],audit_manifest(mutated,old,n=15)['records'])
        self.assertFalse({r['id'] for r in first['records']}&{r['id'] for r in old['records']})
        self.assertTrue(all(set(r)=={'id','conv_id','qa_index','category','split'} for r in first['records']))


if __name__=='__main__':
    unittest.main()
