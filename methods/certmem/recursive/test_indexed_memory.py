from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import refine as core
from indexed_memory import indexed_evidence, index_view, split_fused_index, dedupe_memory
from evaluate_indexed import evaluate
from test_refine import samples


class IndexedMemoryTests(unittest.TestCase):
    def test_fused_split_preserves_provenance_and_only_moves_fact_to_key(self):
        unit={'text':'Facts:\nAlice visited Paris.\nOriginal dialogue:\n[D1:1] Alice: I went to Paris.',
              'kind':'fused_dialogue','sources':['D1:1'],'session':1}
        split=split_fused_index([unit],{})[0]
        self.assertEqual(split['index_text'],'Alice visited Paris.')
        self.assertEqual(split['text'],'[D1:1] Alice: I went to Paris.')
        self.assertEqual(split['sources'],unit['sources'])
        self.assertEqual(split_fused_index([unit],{'read_summary':True})[0]['text'],unit['text'])

    def test_distinct_keys_for_the_same_evidence_are_preserved(self):
        units=[{'text':'evidence','index_text':'color'},{'text':'evidence','index_text':'preference'}]
        self.assertEqual(dedupe_memory(units),units)

    def test_normalized_index_leaves_reader_evidence_exactly_unchanged(self):
        text='(Recorded on: 8 May 2023) [D1:1] Alice: I visited Paris on 7 May 2023 [original relative expression: yesterday]'
        original={'text':text,'sources':['D1:1'],'kind':'raw','session':1}
        result=index_view([original],{'remove_stopwords':True})[0]
        self.assertEqual(result['text'],text)
        self.assertNotIn('D1:1',result['index_text'])
        self.assertIn('Paris',result['index_text'])
        self.assertIn('7 May 2023',result['index_text'])

    def test_key_and_payload_separate_without_losing_source(self):
        sessions=core.session_data(samples()[0])
        facts=[{'text':'Alice visited Paris.','sources':['D1:1'],'session':1,'kind':'fact'}]
        units=indexed_evidence(None,sessions,facts,{'before':0,'anchor':False})
        fact=next(u for u in units if u['kind']=='indexed_evidence')
        self.assertEqual(fact['index_text'],'Alice visited Paris.')
        self.assertIn('Alice: I visited Paris yesterday.',fact['text'])
        self.assertEqual(fact['fact_sources'],['D1:1'])
        self.assertEqual({sid for u in units for sid in u['sources']},{'D1:1','D1:2','D1:3'})
        self.assertNotIn('GOLD_SENTINEL',str(units))

    def test_existing_memory_uses_original_evaluator(self):
        memory={'c':[{'text':'original evidence'}]}
        with patch('refine.evaluate',return_value='original result') as original:
            self.assertEqual(evaluate(None,memory,[],{},20,'baseline','unused'),'original result')
            original.assert_called_once_with(None,memory,[],{},20,'baseline','unused')

    def test_retrieve_by_key_read_payload_and_count_storage(self):
        class BM25:
            def __init__(self,docs): self.docs=docs
            def get_scores(self,words): return [sum(w in d for w in words) for d in self.docs]
        class Runtime:
            ntok=staticmethod(len)
            def encode(self,texts,query=False):
                return np.array([[1.,0.] if 'color' in t.lower() else [0.,1.] for t in texts])
            def generate(self,system,users,max_tokens):
                self.prompt=users[0]
                assert system==core.READER and max_tokens==96
                return ['blue']
        rt=Runtime()
        units=[{'index_text':'INDEX_ONLY color','text':'Anna prefers blue.','sources':['D1:1']},
               {'index_text':'INDEX_ONLY food','text':'Anna likes apples.','sources':['D1:2']}]
        records=[{'id':'c:0','conv_id':'c','qa_index':0,'split':'dev','category':4}]
        data={'c':{'qa':[{'question':'What color?','answer':'blue','category':4,'evidence':['D1:1']}]}}
        with tempfile.TemporaryDirectory() as output, patch.dict(sys.modules,{'rank_bm25':SimpleNamespace(BM25Okapi=BM25)}):
            rows=evaluate(rt,{'c':units},records,data,20,'indexed',output)
        self.assertIn('Anna prefers blue.',rt.prompt)
        self.assertNotIn('INDEX_ONLY',rt.prompt)
        self.assertNotIn('Anna likes apples.',rt.prompt)
        self.assertEqual(rows[0]['official_f1'],1)
        self.assertLessEqual(rows[0]['read_tokens'],20)
        self.assertEqual(rows[0]['memory_tokens'],sum(len(u['text'])+len(u['index_text']) for u in units))


if __name__=='__main__':
    unittest.main()
