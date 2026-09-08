import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from refine import build, f1, make_manifest, pack, parse_facts, quoted_memory, session_data, windows


def samples():
    return [{'sample_id': f'conv-{i}', 'conversation': {
        'speaker_a': 'Alice', 'speaker_b': 'Bob',
        'session_1_date_time': '7 May 2023',
        'session_1': [{'speaker':'Alice', 'text':'I visited Paris yesterday.', 'dia_id':'D1:1'},
                      {'speaker':'Bob', 'text':'Was the museum open?', 'dia_id':'D1:2'},
                      {'speaker':'Alice', 'text':'No, it was closed.', 'dia_id':'D1:3'}]},
        'qa': [{'category': (j % 4)+1, 'question': f'Question {j}',
                'answer':'GOLD_SENTINEL', 'evidence':['D1:3']} for j in range(20)]}
        for i in range(10)]


class FakeRuntime:
    def __init__(self):
        self.prompts = []

    def generate(self, system, users, max_tokens):
        self.prompts.extend([system] + users)
        return ['[D1:1] Alice visited Paris yesterday.\n[D1:2,D1:3] The museum was closed.'] * len(users)


class ProtocolTests(unittest.TestCase):
    def test_manifest_has_100_and_disjoint_conversations(self):
        m = make_manifest(samples())
        self.assertEqual(len(m['records']), 100)
        self.assertEqual(sum(r['split'] == 'dev' for r in m['records']), 70)
        self.assertEqual(len({r['id'] for r in m['records']}), 100)
        dev = {r['conv_id'] for r in m['records'] if r['split'] == 'dev'}
        held = {r['conv_id'] for r in m['records'] if r['split'] == 'holdout'}
        self.assertFalse(dev & held)
        self.assertEqual(m, make_manifest(samples()))

    def test_selection_does_not_depend_on_answer_or_evidence(self):
        data = samples()
        original = make_manifest(data)['records']
        for sample in data:
            for qa in sample['qa']:
                qa['answer'] = 'REPLACED'
                qa['evidence'] = ['D999:999']
        self.assertEqual(original, make_manifest(data)['records'])

    def test_builder_cannot_receive_benchmark_fields(self):
        sample = samples()[0]
        sessions = session_data(sample)
        rt = FakeRuntime()
        baseline = build(rt, sessions, 'session10')
        events = build(rt, sessions, 'events')
        build(rt, sessions, 'audit', events)
        build(rt, sessions, 'dialogue_residual', baseline)
        self.assertNotIn('GOLD_SENTINEL', '\n'.join(rt.prompts))
        self.assertNotIn('Question 0', '\n'.join(rt.prompts))
        mutated = copy.deepcopy(sample)
        mutated['qa'] = [{'question':'DIFFERENT'}]
        self.assertEqual(sessions, session_data(mutated))

    def test_source_ids_must_exist(self):
        session = session_data(samples()[0])[0]
        text = '[D1:1] valid\n[D99:1] fabricated\n[D1:1,D99:1] mixed invalid\nNo source\nNONE'
        units = parse_facts(text, session['turns'], session, 10)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]['sources'], ['D1:1'])

    def test_quoted_memory_keeps_exact_source_and_groups_duplicate_provenance(self):
        sessions = session_data(samples()[0])
        parent = [{'text':'Alice visited Paris.', 'sources':['D1:1'], 'session':1, 'kind':'fact'},
                  {'text':'Alice traveled yesterday.', 'sources':['D1:1'], 'session':1, 'kind':'fact'}]
        result = quoted_memory(sessions,parent)
        self.assertEqual(len(result),1)
        self.assertIn('Alice: I visited Paris yesterday.', result[0]['text'])
        self.assertIn('Alice traveled yesterday.', result[0]['text'])
        self.assertEqual(result[0]['sources'],['D1:1'])
        self.assertNotIn('GOLD_SENTINEL',result[0]['text'])

    def test_context_budget_includes_separator_and_skips_large_units(self):
        units = [{'text':'01234567890'}, {'text':'ab'}, {'text':'cd'}, {'text':'e'}]
        ctx, ids, count = pack(units, [0,1,2,3], len, 6)
        self.assertEqual(ctx, 'ab\n\ncd')
        self.assertEqual(ids, [1,2])
        self.assertEqual(count, 6)
        self.assertLessEqual(len(ctx), 6)

    def test_window_retains_all_source_turns(self):
        s = session_data(samples()[0])[0]
        s['turns'] = [{'id':f'D1:{i}'} for i in range(57)]
        chunks = list(windows([s]))
        self.assertEqual({t['id'] for _, ts in chunks for t in ts}, {t['id'] for t in s['turns']})
        self.assertTrue(all(len(ts) <= 18 for _, ts in chunks))

    def test_official_metric_category_rules(self):
        self.assertEqual(f1('dogs', 'dog', 4), 1)
        self.assertEqual(f1('the museum and park', 'museum park', 4), 1)
        self.assertEqual(f1('Paris', 'Paris; another acceptable response', 3), 1)
        self.assertEqual(f1('Paris', 'Paris, London', 1), 0.5)
        self.assertEqual(f1('Paris, London', 'London, Paris', 1), 1)
        self.assertEqual(f1('', '', 4), 0)
        self.assertEqual(f1('one', '1', 4), 0)
        with self.assertRaises(ValueError):
            f1('unknown', '', 5)


if __name__ == '__main__':
    unittest.main()
