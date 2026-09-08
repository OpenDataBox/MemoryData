import unittest
from provenance_payload import window_for,literal_facts,construct
import refine as core


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.sessions=[{'date':'1 Jan 2024','turns':[{'id':'D1:1','text':'[D1:1] A: I bought a red bicycle.'},
            {'id':'D1:2','text':'[D1:2] B: Nice bicycle.'},{'id':'D1:3','text':'[D1:3] A: It arrived today.'}]}]
        self.turns={t['id']:t for s in self.sessions for t in s['turns']}

    def test_literal_evidence_and_focus_requirement(self):
        w=window_for(self.sessions,['D1:3'],len,1000)
        okay={'text':'A\'s red bicycle arrived.','evidence':[{'source_id':'D1:3','quote':'It arrived today.'},{'source_id':'D1:1','quote':'a red bicycle'}]}
        bad={'text':'Invented','evidence':[{'source_id':'D1:3','quote':'a blue boat'}]}
        neighbor={'text':'B likes it','evidence':[{'source_id':'D1:2','quote':'Nice bicycle.'}]}
        good,rejected=literal_facts(w,{'object':{'facts':[okay,bad,neighbor]}},self.turns)
        self.assertEqual(len(good),1);self.assertEqual(len(rejected),2)

    def test_frozen_keys_modes_cost_and_provenance(self):
        baseline=[{'text':'x'*400,'sources':['D1:3'],'session':1,'kind':'original'}]
        key=core.digest(['D1:3']);fact={'id':'fact_0','text':'A\'s bicycle arrived.','evidence':[{'source_id':'D1:3','quote':'arrived'},{'source_id':'D1:1','quote':'bicycle'}],'supported':True,'deletion_sensitive':False}
        compiled={key:{'facts':[fact]}}
        for mode,count in [('raw',0),('normalized',1),('supported',1),('dependent',0)]:
            memory,d=construct(baseline,self.sessions,compiled,len,mode)
            self.assertEqual(memory[0]['index_text'],baseline[0]['text'])
            self.assertEqual(d['selected_statements'],count)
            self.assertLessEqual(d['storage_tokens'],d['storage_cap'])
            self.assertIn('It arrived today.',memory[0]['text'])
            self.assertEqual('D1:1' in memory[0]['sources'],bool(count))

    def test_oversize_focus_and_payload_fallback(self):
        w=window_for(self.sessions,['D1:1'],len,5)
        self.assertTrue(w['oversized'])
        baseline=[{'text':'small','sources':['D1:1'],'session':1}]
        memory,d=construct(baseline,self.sessions,{core.digest(['D1:1']):{'facts':[]}},len,'dependent')
        self.assertEqual(memory[0]['text'],'small');self.assertEqual(d['fallback_original_payloads'],1)
        with self.assertRaises(ValueError):window_for(self.sessions,['D9:9'],len)


if __name__=='__main__':unittest.main()
