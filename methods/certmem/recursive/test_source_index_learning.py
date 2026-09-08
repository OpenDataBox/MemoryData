from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from source_index_learning import support_mask,loss_gradient,project_cap,optimize_index


class SourceIndexTests(unittest.TestCase):
    def test_gradient_matches_finite_differences_with_multiple_positive_memories(self):
        rng=np.random.default_rng(3)
        docs=rng.normal(size=(3,4));queries=rng.normal(size=(2,4))
        mask=np.array([[1,1,0],[0,1,0]],dtype=bool)
        _,gradient=loss_gradient(docs,queries,mask,.7)
        for index in np.ndindex(docs.shape):
            a,b=docs.copy(),docs.copy()
            a[index]+=1e-6;b[index]-=1e-6
            numeric=(loss_gradient(a,queries,mask,.7)[0]-loss_gradient(b,queries,mask,.7)[0])/2e-6
            self.assertAlmostEqual(numeric,gradient[index],places=6)

    def test_optimizer_improves_source_loss_while_preserving_per_memory_radius(self):
        docs=np.array([[.6,.8],[.8,.6],[0.,1.]])
        queries=np.array([[1.,0.]])
        mask=np.array([[1,0,0]],dtype=bool)
        learned,stats=optimize_index(docs,queries,mask,.15,steps=100,temperature=.1)
        self.assertLess(stats['final_loss'],stats['initial_loss'])
        self.assertLessEqual(np.linalg.norm(learned-docs,axis=1).max(),.150001)
        self.assertTrue(np.allclose(np.linalg.norm(learned,axis=1),1))
        fixed,_=optimize_index(docs,queries,mask,0,steps=5)
        self.assertTrue(np.allclose(fixed,docs))
        self.assertTrue(np.allclose(project_cap(-docs,docs,.1),docs))

    def test_source_support_and_audit_boundary(self):
        units=[{'sources':['a','b']},{'sources':['c']}]
        self.assertEqual(support_mask(units,[{'split':'probe_fit','source_ids':['b']}]).tolist(),[[True,False]])
        with self.assertRaises(ValueError):
            support_mask(units,[{'split':'probe_audit','source_ids':['b']}])
        with self.assertRaises(ValueError):
            support_mask(units,[{'split':'probe_fit','source_ids':['missing']}])


if __name__=='__main__':
    unittest.main()
