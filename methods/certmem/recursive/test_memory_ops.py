import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from memory_ops import clean_cards, fuse_evidence, apply_operation, salient_dialogue
from continuous import dialogue_blocks
from refine import session_data
from test_refine import samples


class Runtime:
    ntok=staticmethod(len)


class MemoryOperationsTests(unittest.TestCase):
    def test_cards_remove_unanswered_questions_and_duplicate_values(self):
        texts=['Alice | activity | went to Paris yesterday','Alice | experience | went to Paris yesterday',
               'Bob | reply | Where did you go?','Bob | reaction | Wow, cool!',
               'Bob | activity | went to Paris yesterday']
        units=[{'text':text,'sources':['D1:1'],'session':1,'kind':'extractive_fact'} for text in texts]
        result=clean_cards(units)
        self.assertEqual(len(result),2)
        self.assertTrue(any('Alice' in u['text'] for u in result))
        self.assertTrue(any('Bob' in u['text'] for u in result))

    def test_fusion_preserves_all_source_text_and_fact_provenance(self):
        sessions=session_data(samples()[0])
        parent=[{'text':'Alice visited Paris.','sources':['D1:1'],'session':1,'kind':'fact'},
                {'text':'Museum was closed.','sources':['D1:2','D1:3'],'session':1,'kind':'fact'}]
        old=copy.deepcopy(parent)
        result=fuse_evidence(Runtime(),sessions,parent,{'size':2,'overlap':1,'unit_budget':1000})
        self.assertEqual(parent,old)
        combined='\n'.join(u['text'] for u in result)
        for turn in sessions[0]['turns']:
            self.assertIn(turn['text'],combined)
        for fact in parent:
            self.assertIn(fact['text'],combined)
        self.assertEqual({s for u in result for s in u['sources']},{'D1:1','D1:2','D1:3'})

    def test_fusion_overflow_keeps_fact_as_residual(self):
        sessions=session_data(samples()[0])
        fact={'text':'x'*500,'sources':['D1:1'],'session':1,'kind':'fact'}
        result=fuse_evidence(Runtime(),sessions,[fact],{'size':2,'overlap':1,'unit_budget':20})
        self.assertIn(fact,result)

    def test_selective_anchor_preserves_non_temporal_record_bytes(self):
        sessions=session_data(samples()[0])
        units=dialogue_blocks(sessions,size=1)
        result=apply_operation(Runtime(),sessions,units,{'op':'selective_anchor'})
        self.assertEqual(result[1],units[1])
        self.assertEqual(result[2],units[2])
        self.assertNotEqual(result[0]['text'],units[0]['text'])
        self.assertIn('6 May 2023',result[0]['text'])

    def test_salience_rejects_invented_source_ids_and_preserves_substantive_images(self):
        sessions=session_data(samples()[0])
        sessions[0]['turns'][0]['body']+=' [image: museum ticket]'
        class Classifier(Runtime):
            def generate(self,system,users,max_tokens):
                assert 'GOLD_SENTINEL' not in '\n'.join(users)
                return ['DROP [D999:1]\nDROP [D1:1]\nDROP [D1:2]']*len(users)
        result=salient_dialogue(Classifier(),sessions,[],{'include_facts':False})
        sources={sid for unit in result for sid in unit['sources']}
        self.assertIn('D1:1',sources)
        self.assertIn('D1:3',sources)
        self.assertIn('D1:2',sources)  # Its question is retained to interpret the reply.
        self.assertNotIn('D999:1',sources)


if __name__=='__main__':
    unittest.main()
