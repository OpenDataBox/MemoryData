from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from evidence_utility import (answer_token_request,extract_answer_logprob,subset_jobs,
                             subset_analysis,select_probes)


class UtilityTests(unittest.TestCase):
    def test_actual_target_token_is_selected_not_first_logprob_entry(self):
        output=SimpleNamespace(prompt_token_ids=[4,5,6],prompt_logprobs=[None,
                {999:SimpleNamespace(logprob=-99),5:SimpleNamespace(logprob=-1)},
                {6:SimpleNamespace(logprob=-3)}])
        result=extract_answer_logprob(output,1,[5,6])
        self.assertEqual(result['mean_logprob'],-2)
        with self.assertRaises(ValueError):
            extract_answer_logprob(output,1,[5,7])

    def test_target_boundary_uses_explicit_token_concatenation(self):
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:[1,2],
                                  encode=lambda text,**k:[3,4])
        request,start,target=answer_token_request(tokenizer,'q','a')
        self.assertEqual(request['prompt_token_ids'],[1,2,3,4])
        self.assertEqual(start,2)
        self.assertEqual(target,[3,4])

    def test_all_pairs_are_considered_even_when_both_singletons_are_useless(self):
        probe={'candidate_context_ids':['a','b','c'],'source_ids':['b'],
               'recorded_date':'date','question':'q','answer':'x'}
        turns={sid:{'text':sid} for sid in ('a','b','c')}
        jobs,skip=subset_jobs(probe,turns,len)
        self.assertIsNone(skip)
        self.assertEqual(len([j for j in jobs if j['kind']=='pair']),3)
        for job in jobs:
            value=-5
            if job['kind']=='full' or job['source_ids']==['a','b']:
                value=-0.2
            job['score']={'mean_logprob':value}
        analysis=subset_analysis(jobs)
        self.assertEqual(analysis['minimum_sources'],['a','b'])
        self.assertTrue(analysis['has_preserving_pair_without_preserving_single'])
        self.assertEqual(analysis['single_policy_sources'],['a','b','c'])

    def test_overbudget_reference_is_rejected_without_silent_truncation(self):
        probe={'candidate_context_ids':['a'],'source_ids':['a'],'recorded_date':'date','question':'q','answer':'x'}
        jobs,reason=subset_jobs(probe,{'a':{'text':'long'}},len,budget=1)
        self.assertFalse(jobs)
        self.assertEqual(reason,'full_source_window_exceeds_read_budget')

    def test_pilot_selection_is_independent_of_question_and_answer_content(self):
        rows=[{'id':str(i),'conv_id':'c','split':'probe_fit' if i<8 else 'probe_audit',
               'question':'q','answer':'a'} for i in range(12)]
        first=select_probes({'records':rows})
        changed=[dict(p,question='mutated',answer='mutated') for p in rows]
        self.assertEqual([p['id'] for p in first],[p['id'] for p in select_probes({'records':changed})])


if __name__=='__main__':
    unittest.main()
