import unittest
import numpy as np
import refine as core
from coverage_routes import harvest,select,construct


class CoverageTests(unittest.TestCase):
    def test_marginal_coverage_updates_and_source_exclusion(self):
        candidates=[{'id':str(i),'cost':1,'unit':{'sources':[s]},'deletion_sensitive':i!=1} for i,s in enumerate(['a','b','c','a'])]
        geom={'similarities':np.array([[1,.99,0,1],[.99,1,0,1],[0,0,1,0],[1,1,0,1]]),'initial_coverage':np.zeros(4)}
        ids,stats=select(candidates,geom,2,'supported')
        self.assertIn(2,ids);self.assertEqual(len(ids),2)
        self.assertFalse(0 in ids and 3 in ids);self.assertEqual(stats['spent'],2)
        ids,_=select(candidates,geom,4,'dependent');self.assertNotIn(1,ids)

    def test_harvest_only_supported_self_contained_routes_and_preserves_base(self):
        base=[{'text':'source body','sources':['D1:1'],'session':1}]
        fact=lambda fid,sup,cite:{'id':fid,'text':fid,'supported':sup,'deletion_sensitive':True,'evidence':[{'source_id':cite,'quote':'body'}]}
        source={core.digest(['D1:1']):{'facts':[fact('yes',True,'D1:1'),fact('no',False,'D1:1'),fact('neighbor',True,'D1:2')]}}
        candidates=harvest(base,source,len);self.assertEqual(len(candidates),1)
        geom={'similarities':np.ones((1,1)),'initial_coverage':np.zeros(1)}
        memory,d=construct(base,candidates,geom,len,100,'supported')
        self.assertEqual(memory[:1],base);self.assertEqual(memory[1]['text'],base[0]['text'])
        self.assertEqual(d['added_routes'],1);self.assertLessEqual(d['storage_tokens'],d['storage_cap'])


if __name__=='__main__':unittest.main()
