import unittest
import numpy as np
from transfer_index_alignment import normalized,support_centroids,fit_rotation,apply_rotation,training_conversations


class AlignmentTests(unittest.TestCase):
    def test_rotation_preserves_geometry_and_improves_small_step_alignment(self):
        rng=np.random.default_rng(421)
        documents=normalized(rng.normal(size=(40,24)))
        supports=documents[:8]
        queries=normalized(supports+.5*rng.normal(size=supports.shape))
        transform,details=fit_rotation(supports,queries,.1)
        result,stats=apply_rotation(documents,transform,.1)
        self.assertGreater(details['source_mean_cosine_after'],details['source_mean_cosine_before'])
        np.testing.assert_allclose(result@result.T,documents@documents.T,atol=2e-7)
        self.assertLessEqual(stats['maximum_document_distance'],.100001)
        self.assertLessEqual(stats['maximum_pairwise_inner_product_error'],1e-6)

    def test_zero_and_degenerate_pairs_do_not_change_vectors(self):
        docs=np.eye(6)
        for radius,query in [(0,np.roll(docs,1,axis=0)),(.3,docs)]:
            transform,_=fit_rotation(docs,query,radius)
            result,_=apply_rotation(docs,transform,radius)
            np.testing.assert_allclose(result,docs,atol=1e-7)
        with self.assertRaises(ValueError):fit_rotation(docs,docs,2)

    def test_positive_centroids_and_target_exclusion(self):
        docs=np.eye(4);positive=np.array([[1,1,0,0],[0,0,1,0]],dtype=bool)
        np.testing.assert_allclose(support_centroids(docs,positive),[[2**-.5,2**-.5,0,0],[0,0,1,0]])
        self.assertEqual(training_conversations('a',['a','b','c'],'transfer'),['b','c'])
        self.assertEqual(training_conversations('a',['a','b','c'],'within'),['a'])
        with self.assertRaises(ValueError):training_conversations('a',['a'],'transfer')
        with self.assertRaises(ValueError):support_centroids(docs,np.zeros((2,4),dtype=bool))


if __name__=='__main__':unittest.main()
