"""Learn stored document vectors from source-derived query/support pairs.

The encoder and reader are fixed. Each normalized document vector is constrained
to a spherical cap around its original embedding. This is projected Adam on a
multi-positive retrieval loss, not a guarantee of unseen-question preservation.
"""
import numpy as np


def support_mask(units,probes):
    if any(p['split']!='probe_fit' for p in probes):
        raise ValueError('Only source-fit probes may train memory vectors')
    mask=np.asarray([[bool(set(u['sources'])&set(p['source_ids'])) for u in units] for p in probes],dtype=bool)
    if mask.ndim!=2 or not len(probes) or not np.all(mask.any(axis=1)):
        raise ValueError('Each source query needs at least one source-covering memory')
    return mask


def loss_gradient(vectors,queries,positive,temperature=.05):
    if temperature<=0 or positive.shape!=(len(queries),len(vectors)) or not np.all(positive.any(axis=1)):
        raise ValueError('Invalid contrastive retrieval objective')
    logits=queries@vectors.T/temperature
    top=logits.max(axis=1,keepdims=True)
    weights=np.exp(logits-top)
    denom=weights.sum(axis=1,keepdims=True)
    masked=np.where(positive,logits,-np.inf)
    pos_top=masked.max(axis=1,keepdims=True)
    pos_weights=np.exp(masked-pos_top)
    pos_denom=pos_weights.sum(axis=1,keepdims=True)
    loss=np.mean(top+np.log(denom)-pos_top-np.log(pos_denom))
    gradient=(weights/denom-pos_weights/pos_denom).T@queries/(len(queries)*temperature)
    return float(loss),gradient


def project_cap(proposal,anchor,radius):
    if not 0<=radius<2:
        raise ValueError('Radius must lie in [0,2)')
    norms=np.linalg.norm(proposal,axis=1,keepdims=True)
    direction=np.divide(proposal,norms,out=anchor.copy(),where=norms>1e-12)
    cosine=np.clip(np.sum(direction*anchor,axis=1,keepdims=True),-1,1)
    minimum=1-radius*radius/2
    outside=cosine[:,0]<minimum
    tangent=direction-cosine*anchor
    tangent_norm=np.linalg.norm(tangent,axis=1,keepdims=True)
    tangent=np.divide(tangent,tangent_norm,out=np.zeros_like(tangent),where=tangent_norm>1e-12)
    boundary=minimum*anchor+np.sqrt(max(0,1-minimum*minimum))*tangent
    # At the exact antipode the closest cap point is not uniquely determined.
    boundary[tangent_norm[:,0]<=1e-12]=anchor[tangent_norm[:,0]<=1e-12]
    direction[outside]=boundary[outside]
    if radius==0:
        return anchor.copy()
    return direction


def optimize_index(original,queries,positive,radius,steps=200,learning_rate=.01,temperature=.05):
    original=np.asarray(original,dtype=np.float64)
    queries=np.asarray(queries,dtype=np.float64)
    if original.ndim!=2 or queries.ndim!=2 or original.shape[1]!=queries.shape[1]:
        raise ValueError('Vector dimensions differ')
    norms=np.linalg.norm(original,axis=1,keepdims=True)
    if np.any(norms<=1e-12) or not np.isfinite(original).all() or not np.isfinite(queries).all():
        raise ValueError('Invalid source embeddings')
    anchor=original/norms
    vectors=anchor.copy()
    first,second=np.zeros_like(vectors),np.zeros_like(vectors)
    trace=[]
    initial,_=loss_gradient(vectors,queries,positive,temperature)
    trace.append({'step':0,'loss':initial})
    for step in range(1,steps+1):
        loss,gradient=loss_gradient(vectors,queries,positive,temperature)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError('Nonfinite index optimization')
        first=.9*first+.1*gradient
        second=.999*second+.001*gradient*gradient
        proposal=vectors-learning_rate*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
        vectors=project_cap(proposal,anchor,radius)
        if step%50==0 or step==steps:
            current,_=loss_gradient(vectors,queries,positive,temperature)
            trace.append({'step':step,'loss':current})
    result=vectors.astype(np.float32)
    distance=np.linalg.norm(result.astype(np.float64)-anchor,axis=1)
    assert distance.max()<=radius+1e-6
    assert np.max(np.abs(np.linalg.norm(result,axis=1)-1))<1e-6
    return result,{'radius':radius,'steps':steps,'learning_rate':learning_rate,'temperature':temperature,
                   'training_queries':len(queries),'documents':len(original),'trace':trace,
                   'initial_loss':initial,'final_loss':loss_gradient(result.astype(np.float64),queries,positive,temperature)[0],
                   'maximum_distance':float(distance.max()),'mean_distance':float(distance.mean()),
                   'stored_dtype':str(result.dtype),'stored_bytes':result.nbytes}
