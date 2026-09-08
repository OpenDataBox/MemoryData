import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from identity_consolidation import consolidate,identity


class ConsolidationTests(unittest.TestCase):
    def test_duplicate_evidence_is_read_once_and_keys_are_combined(self):
        units=[{'text':'Evidence','sources':['a'],'session':1},
               {'text':'Other evidence','sources':['b'],'session':1},
               {'text':'Evidence','index_text':'Question?','sources':['a'],'session':1,'source_probe_id':'p'},
               {'text':'Evidence','index_text':'Second?','sources':['a'],'session':1,'source_probe_id':'q'}]
        saved=copy.deepcopy(units)
        output,stats=consolidate(units,len,'union')
        self.assertEqual(len(output),2)
        self.assertEqual(output[0]['index_text'],'Evidence\n\nQuestion?\n\nSecond?')
        self.assertEqual(output[0]['source_probe_ids'],['p','q'])
        self.assertEqual(stats['removed_duplicate_payloads'],2)
        self.assertEqual(units,saved)
        self.assertEqual(consolidate(output,len,'union')[0],output)
        self.assertEqual(consolidate(units,len,'cues')[0][0]['index_text'],'Question?\n\nSecond?')
        self.assertNotIn('index_text',consolidate(units,len,'payload')[0][0])

    def test_similar_text_or_different_provenance_must_not_merge(self):
        units=[{'text':'Same','sources':['a'],'session':1},
               {'text':'Same','sources':['b'],'session':1},
               {'text':'Same','sources':['a'],'session':2},
               {'text':'Same plus','sources':['a'],'session':1}]
        result,stats=consolidate(units,len,'union')
        self.assertEqual(result,units)
        self.assertEqual(stats['removed_duplicate_payloads'],0)
        self.assertEqual(len({identity(u) for u in result}),4)


if __name__=='__main__':
    unittest.main()
