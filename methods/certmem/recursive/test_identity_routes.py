import unittest
import numpy as np
from identity_routes import construct,common_candidates


class IdentityRouteTests(unittest.TestCase):
    def fixture(self):
        baseline=[{'text':'original source body','sources':['D1:1'],'session':1},
                  {'text':'another body','index_text':'preexisting specific key','sources':['D1:2'],'session':1}]
        candidates=[]
        for i,parent in enumerate(baseline):
            route={**parent,'index_text':'fact '+str(i),'parent_index':i,'source_group':str(i),'source_fact_id':str(i)}
            candidates.append({'id':str(i),'unit':route,'cost':999,'deletion_sensitive':True})
        return baseline,candidates,{'similarities':np.eye(2),'initial_coverage':np.zeros(2)}

    def test_controls_share_facts_and_budget_but_read_identity_differs(self):
        baseline,candidates,geom=self.fixture();selected=[]
        for mode in ('copy','union','replace'):
            memory,details=construct(baseline,candidates,geom,len,200,mode)
            selected.append(details['selected_ids'])
            self.assertEqual([u['text'] for u in memory[:2]],[u['text'] for u in baseline])
            self.assertEqual(len(memory),4 if mode=='copy' else 2)
            self.assertLessEqual(details['actual_added_tokens'],details['spent'])
            self.assertLessEqual(details['storage_tokens'],details['storage_cap'])
            if mode=='union':self.assertEqual(memory[1]['index_text'],'preexisting specific key\nfact 1')
        self.assertEqual(selected[0],selected[1]);self.assertEqual(selected[1],selected[2])
        self.assertNotIn('index_text',baseline[0]);self.assertNotIn('route_patch',baseline[1])

    def test_common_cost_uses_expensive_control_and_empty_budget_is_identity(self):
        baseline,candidates,geom=self.fixture();paired=common_candidates(baseline,candidates,len)
        for item in paired:self.assertEqual(item['cost'],max(1,*item['mode_incremental_costs'].values()))
        for mode in ('copy','union','replace'):
            memory,details=construct(baseline,candidates,geom,len,0,mode)
            self.assertEqual(memory,baseline);self.assertEqual(details['spent'],0)


if __name__=='__main__':unittest.main()
