"""Rigid memory-index alignment learned from source-derived support centroids.

One Cayley rotation, confined to the span of source query/support pairs, preserves
all document-document inner products. Cross-conversation fitting excludes every
source question of the target conversation. This is not a novelty claim for
orthogonal alignment, Cayley transforms, or transfer learning.
"""
import numpy as np


def normalized(values):
    values=np.asarray(values,dtype=np.float64)
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if values.ndim!=2 or not np.isfinite(values).all() or np.any(norms<=1e-12):
        raise ValueError('Invalid vectors')
    return values/norms


def support_centroids(documents,positive):
    documents=normalized(documents)
    if positive.ndim!=2 or positive.shape[1]!=len(documents) or not np.all(positive.any(axis=1)):
        raise ValueError('Every source query needs a positive support centroid')
    return normalized(positive.astype(np.float64)@documents/positive.sum(axis=1,keepdims=True))


def fit_rotation(supports,queries,radius):
    supports,queries=normalized(supports),normalized(queries)
    if supports.shape!=queries.shape or not 0<=radius<2:
        raise ValueError('Invalid source alignment')
    # An SVD basis removes dependent directions; the complement remains identity.
    _,singular,vt=np.linalg.svd(np.concatenate([supports,queries]),full_matrices=False)
    basis=vt[singular>singular[0]*1e-10].T
    db,qb=supports@basis,queries@basis
    skew=(qb.T@db-db.T@qb)/len(supports)
    opnorm=np.linalg.norm(skew,ord=2)
    eye=np.eye(len(skew))
    if radius==0 or opnorm<1e-12:
        rotation=eye
    else:
        gamma=radius/np.sqrt(4-radius*radius)
        scaled=gamma*skew/opnorm
        rotation=np.linalg.solve(eye-scaled,eye+scaled)
    assert np.max(np.abs(rotation.T@rotation-eye))<1e-10
    assert np.linalg.norm(rotation-eye,ord=2)<=radius+1e-10
    aligned=supports+(supports@basis)@(rotation.T-eye)@basis.T
    return {'basis':basis,'rotation':rotation},{'radius':radius,'training_queries':len(supports),
        'subspace_dimension':basis.shape[1],'skew_operator_norm':float(opnorm),
        'rotation_operator_distance':float(np.linalg.norm(rotation-eye,ord=2)),
        'source_mean_cosine_before':float(np.mean(np.sum(supports*queries,axis=1))),
        'source_mean_cosine_after':float(np.mean(np.sum(aligned*queries,axis=1)))}


def apply_rotation(documents,transform,radius):
    original=normalized(documents)
    basis,rotation=transform['basis'],transform['rotation']
    result=(original+(original@basis)@(rotation.T-np.eye(len(rotation)))@basis.T).astype(np.float32)
    distance=np.linalg.norm(result.astype(np.float64)-original,axis=1)
    assert distance.max()<=radius+1e-6
    assert np.max(np.abs(np.linalg.norm(result,axis=1)-1))<1e-6
    error=float(np.max(np.abs(result.astype(np.float64)@result.astype(np.float64).T-original@original.T)))
    assert error<1e-6
    return result,{'maximum_document_distance':float(distance.max()),'mean_document_distance':float(distance.mean()),
                   'maximum_pairwise_inner_product_error':error,'stored_dtype':str(result.dtype),'stored_bytes':result.nbytes}


def training_conversations(target,available,scope):
    if target not in available or scope not in ('within','transfer'):
        raise ValueError('Invalid conversation split')
    chosen=[target] if scope=='within' else sorted(cid for cid in available if cid!=target)
    if not chosen:
        raise ValueError('No other conversation for transfer fitting')
    return chosen
