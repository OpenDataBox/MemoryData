from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from portable_parent import augment,generate_source_probes


class PortableTests(unittest.TestCase):
    def test_audit_probe_cannot_enter_parent_constructor(self):
        with self.assertRaises(ValueError):
            augment(None,[],[],[{'split':'probe_audit'}])

    def test_generated_probe_must_ground_to_source_even_when_ids_are_missing(self):
        sessions={'new-conversation':[{'num':1,'date':'1 January 2020','turns':[
            {'id':'D1:1','body':'I own a red bicycle.','speaker':'Alex','text':'[D1:1] Alex: I own a red bicycle.'}]}]}
        seen=[]
        def generate(system,users,max_tokens):
            seen.extend(users)
            return ['Q: What does Alex own? A: a red bicycle\nQ: What is the pet name? A: Unsupported']
        rt=SimpleNamespace(generate=generate)
        dump,pool=generate_source_probes(rt,sessions,{'window':10,'overlap':2,'max_tokens':1400,'prompt':'Source facts only'})
        self.assertEqual(len(pool['records']),1)
        self.assertEqual(pool['records'][0]['answer'],'a red bicycle')
        self.assertEqual(pool['records'][0]['source_ids'],['D1:1'])
        self.assertTrue(all('Unsupported' not in user for user in seen))
        self.assertEqual(dump['records'][0]['turn_ids'],['D1:1'])


if __name__=='__main__':
    unittest.main()
