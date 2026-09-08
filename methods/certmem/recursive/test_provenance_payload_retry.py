import json
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
import provenance_payload_v2 as module


class RetryTests(unittest.TestCase):
    def test_only_incomplete_json_retried_and_all_attempts_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            def generate(prompts,params,use_tqdm):
                calls.append((params.max_tokens,len(prompts)))
                text='{"facts":[' if params.max_tokens==768 else '{"facts":[]}'
                return [SimpleNamespace(outputs=[SimpleNamespace(text=text,finish_reason='length' if params.max_tokens==768 else 'stop')]) for _ in prompts]
            rt=SimpleNamespace(cache=Path(folder),model_meta={'revision':'test'},args=SimpleNamespace(seed=1),
                ntok=len,tok=SimpleNamespace(apply_chat_template=lambda *a,**k:'source'),llm=SimpleNamespace(generate=generate))
            initial=[{'object':None,'raw':'{"facts":[','finish_reason':'length'},
                     {'object':{'facts':[]},'raw':'{"facts":[]}','finish_reason':'stop'},
                     {'object':None,'raw':'','finish_reason':'oversized_source'}]
            windows=[{'oversized':False,'user':'one'},{'oversized':False,'user':'two'},{'oversized':True,'user':None}]
            vllm=ModuleType('vllm');vllm.SamplingParams=lambda **k:SimpleNamespace(**k)
            sampling=ModuleType('vllm.sampling_params');sampling.GuidedDecodingParams=lambda **k:k
            with patch.dict('sys.modules',{'vllm':vllm,'vllm.sampling_params':sampling}),patch.object(module,'_initial_generate',side_effect=lambda *a:copy.deepcopy(initial)):
                values=module.generate(rt,windows)
                self.assertEqual(calls,[(768,1),(1536,1)])
                self.assertEqual([a['max_tokens'] for a in values[0]['attempts']],[384,768,1536])
                self.assertEqual(len(values[1]['attempts']),1);self.assertEqual(values[1]['raw'],initial[1]['raw'])
                self.assertEqual(values[2]['attempts'],[]);self.assertTrue(values[0]['complete_json'])
                calls.clear()
                again=module.generate(rt,windows)
                self.assertEqual(calls,[]);self.assertEqual(values,again)


if __name__=='__main__':unittest.main()
