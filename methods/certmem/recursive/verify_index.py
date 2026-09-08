"""Replay stored-vector optimization and actual retrieval from frozen artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi
import refine as core
from source_index_learning import optimize_index,loss_gradient,support_mask
from budgeted_evidence import storage_cost
from tokenizer_validation import load_counter
from verify_identity import changes
from transfer_index_alignment import support_centroids,fit_rotation,apply_rotation,training_conversations


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path)
    ap.add_argument('--run',default='source_index_v1');ap.add_argument('--out-prefix',default='INDEX');args=ap.parse_args()
    root=args.root;run=root/args.run/'run';p=read(run/'protocol.json')
    rigid=all(r.get('method')=='rigid_cayley_alignment' for r in p['recipes'])
    for name,value in p['source_sha256'].items():assert sha(run.parent/'source'/name)==value
    ntok,meta=load_counter(root/'tokenizer_reference/tokenizer_reference')
    assert meta['model_meta']==p['environment']['models'][p['args']['model']]
    cids=sorted({r['conv_id'] for r in p['records']})
    base={cid:read(run/'base_memories'/f'{cid}.json') for cid in cids}
    assert core.digest(base)==p['baseline_memory_sha256']
    assert base=={cid:read(root/'parent_evidence_v1/run/memories/s_parent_single_2000'/f'{cid}.json') for cid in cids}
    training=sorted([read(f) for f in (root/'evidence_utility_fit_v1/runs/evidence_utility_fit_v1/items').glob('*.json')
                      if 'analysis' in read(f) and read(f)['analysis']['source_dependent'] and read(f)['generations']['full']['source_answer_f1']>=.8],key=lambda r:r['id'])
    a=read(root/'crossview_refinement_v1/run/query_views/fit_views.json');ids={r['id'] for r in a}
    q0=[r for r in training if r['id'] in ids]
    assert all(r['split']=='probe_fit' for r in training+a) and len(training)==97 and len(q0)==len(a)==75
    assert core.digest(training)==p['source_training_digest'] and core.digest(q0)==p['source_q0_digest'] and core.digest(a)==p['fit_view_digest']
    original={cid:np.load(run/'indexes/baseline'/f'{cid}.npy') for cid in cids}
    costs={cid:sum(storage_cost(u,ntok) for u in base[cid]) for cid in cids}
    data={}
    for cid in cids:
        assert original[cid].dtype==np.float32 and original[cid].shape==(len(base[cid]),1024)
        data[cid]=np.load(run/'training'/f'{cid}.npz')
        probes=[r for r in training if r['conv_id']==cid];details=read(run/'training'/f'{cid}.json')
        assert details=={'source_probe_ids':[r['id'] for r in probes],'query_texts_sha256':core.digest([r['question'] for r in probes]),'uses_fit_view_a':False}
        assert np.array_equal(data[cid]['positive'],support_mask(base[cid],probes))
    training_lock=read(run/'training_locked.json')
    assert training_lock['gradient_training_used_only_q0']
    vectors={};maximum_replay_error=0;traces=[]
    for recipe in p['recipes']:
        name=recipe['name'];vectors[name]={}
        print(json.dumps({'verification':'replay_index_construction','name':name}),flush=True)
        for cid in cids:
            assert read(run/'memories'/name/f'{cid}.json')==base[cid]
            v=np.load(run/'indexes'/name/f'{cid}.npy');vectors[name][cid]=v
            assert sha(run/'indexes'/name/f'{cid}.npy')==training_lock['all_indexes_sha256'][name][cid]
            d=read(run/'construction'/name/f'{cid}.json')
            assert d=={'storage_tokens':costs[cid],'storage_cap':costs[cid],'dense_index_bytes':v.nbytes,'text_memory_unchanged':True}
            assert v.dtype==np.float32 and v.shape==original[cid].shape and v.nbytes==original[cid].nbytes
            if rigid:
                train_cids=training_conversations(cid,cids,recipe['scope'])
                supports=np.concatenate([support_centroids(original[k],data[k]['positive']) for k in train_cids])
                queries=np.concatenate([data[k]['queries'] for k in train_cids])
                transform,fit_details=fit_rotation(supports,queries,recipe['radius'])
                replay,apply_details=apply_rotation(original[cid],transform,recipe['radius'])
                stats={**fit_details,**apply_details,'scope':recipe['scope'],'target_conversation':cid,
                       'training_conversations':train_cids,'training_probe_ids':[r['id'] for k in train_cids for r in training if r['conv_id']==k],
                       'uses_target_questions':cid in train_cids}
                stored=np.load(run/'transforms'/name/f'{cid}.npz')
                mapped,_=apply_rotation(original[cid],stored,recipe['radius'])
                assert np.max(np.abs(mapped.astype(np.float64)-v.astype(np.float64)))<1e-6
                assert np.max(np.abs(stored['basis'].T@stored['basis']-np.eye(stored['basis'].shape[1])))<1e-9
                assert np.max(np.abs(stored['rotation'].T@stored['rotation']-np.eye(stored['rotation'].shape[1])))<1e-9
                if recipe['scope']=='transfer':assert cid not in train_cids and len(train_cids)==6
            else:
                replay,stats=optimize_index(original[cid],data[cid]['queries'],data[cid]['positive'],recipe['radius'],steps=200,learning_rate=.01,temperature=.05)
            error=float(np.max(np.abs(replay.astype(np.float64)-v.astype(np.float64))))
            maximum_replay_error=max(maximum_replay_error,error)
            assert error<1e-6
            saved=read(run/'optimizer'/name/f'{cid}.json')
            assert stats.keys()==saved.keys()
            for k in stats:
                if isinstance(stats[k],float):assert abs(stats[k]-saved[k])<1e-6
                elif k=='trace':
                    assert len(stats[k])==len(saved[k])
                    for x,y in zip(stats[k],saved[k]):assert x['step']==y['step'] and abs(x['loss']-y['loss'])<1e-6
                else:assert stats[k]==saved[k]
            anchor=original[cid].astype(np.float64);anchor/=np.linalg.norm(anchor,axis=1,keepdims=True)
            assert np.linalg.norm(v-anchor,axis=1).max()<=recipe['radius']+1e-6
            assert np.max(np.abs(np.linalg.norm(v,axis=1)-1))<1e-6
            traces.append({'name':name,'conv_id':cid,
                           'objective':'support_query_cosine' if rigid else 'retrieval_loss',
                           'initial':saved['source_mean_cosine_before'] if rigid else saved['initial_loss'],
                           'final':saved['source_mean_cosine_after'] if rigid else saved['final_loss'],
                           'max_distance':saved['maximum_document_distance'] if rigid else saved['maximum_distance']})
    query_maps={}
    for split,filename in [('dev','reference_replay_dev_items.json'),('q0','source_baseline_source_fit_items.json'),('fit_a','fit_baseline_source_view_fit_items.json')]:
        rows=read(run/filename);query_maps[split]={}
        for cid in cids:
            selected=[r for r in rows if r['conv_id']==cid];metaq=read(run/'evaluation_queries'/split/f'{cid}.json')
            assert metaq['ids']==[r['id'] for r in selected]
            assert metaq['cache_key']==core.digest([p['environment']['models'][p['args']['embed_model']],[r['question'] for r in selected],True,p['args']['embed_batch_size'],'oom_backoff_v1'])
            q=np.load(run/'evaluation_queries'/split/f'{cid}.npy')
            assert q.shape==(len(selected),1024)
            query_maps[split].update({r['id']:vec for r,vec in zip(selected,q)})
            if split=='q0':
                probes=[r for r in training if r['conv_id']==cid];index={r['id']:i for i,r in enumerate(probes)}
                for r,v in zip(selected,q):assert np.max(np.abs(v-data[cid]['queries'][index[r['id']]]))<1e-6
    def ranks(scores):
        order=np.argsort(-scores,kind='stable');result=np.empty(len(order),dtype=np.int64);result[order]=np.arange(len(order));return result
    bm25={cid:BM25Okapi([core.lexical(u.get('index_text',u['text'])) or ['_empty'] for u in base[cid]]) for cid in cids}
    contexts=0
    def check_rows(filename,indexes,split,contract=None):
        nonlocal contexts
        rows=read(run/filename)
        for row in rows:
            cid=row['conv_id'];units=base[cid];query=query_maps[split][row['id']]
            ds=indexes[cid]@query;ss=np.asarray(bm25[cid].get_scores(core.lexical(row['question'])))
            fused=1/(60+ranks(ds))+1/(60+ranks(ss));order=np.argsort(-fused,kind='stable')[:min(len(units),120)]
            context,hits,tokens=core.pack(units,order,ntok,2048)
            assert row['context']==context and row['read_tokens']==tokens==ntok(context)<=2048
            assert row['source_ids']==sorted({s for i in hits for s in units[i]['sources']})
            assert row['memory_tokens']==costs[cid] and row['dense_index_bytes']==indexes[cid].nbytes
            assert abs(core.f1(row['prediction'],row['gold'],row['category'])-row['official_f1'])<1e-12
            if contract is not None:
                source={r['id']:r for r in (q0 if split=='q0' else a)}[row['id']]
                assert row['question']==source['question'] and row['gold']==source['generations']['full']['text']
                assert row['split']==('source_fit' if split=='q0' else 'source_view_fit') and row['category']==4
                c=contract[row['id']]
                assert c['prediction']==row['prediction'] and c['context_tokens']==row['read_tokens']
                assert c['preserved']==bool(re.fullmatch(r'EQUIVALENT[.!]?',c['verdict'].strip().upper()))
            contexts+=1
        assert len(rows)==(70 if split=='dev' else 75) and len({r['id'] for r in rows})==len(rows)
        print(json.dumps({'verification':'retrieval_replayed','file':filename,'checked_total':contexts}),flush=True)
        return rows
    old=read(run/'original_reference_dev_items.json');new=check_rows('reference_replay_dev_items.json',original,'dev')
    keys=('id','question','context','prediction','source_ids','read_tokens','memory_tokens','official_f1')
    assert [{k:r[k] for k in keys} for r in old]==[{k:r[k] for k in keys} for r in new]
    contract=read(run/'source_baseline_contract.json')
    check_rows('source_baseline_source_fit_items.json',original,'q0',contract['q0'])
    check_rows('fit_baseline_source_view_fit_items.json',original,'fit_a',contract['fit_a'])
    trials={};feasible=[]
    for recipe in p['recipes']:
        name=recipe['name'];t=read(run/'source_trials'/f'{name}.json');trials[name]=t
        check_rows(f'{name}_source_fit_items.json',vectors[name],'q0',t['q0_contract'])
        check_rows(f'{name}_source_view_fit_items.json',vectors[name],'fit_a',t['fit_a_contract'])
        dq,da=changes(contract['q0'],t['q0_contract']),changes(contract['fit_a'],t['fit_a_contract'])
        nq,na=[len(d['repairs'])-len(d['regressions']) for d in (dq,da)]
        expected={'name':name,'radius':recipe['radius'],'q0':dq,'fit_a':da,'net_q0':nq,'net_a':na,'worst_view_gain':min(nq,na),'feasible':nq>0 and na>0}
        assert t['result']==expected
        if expected['feasible']:feasible.append(expected)
        check_rows(f'{name}_dev_items.json',vectors[name],'dev')
    choice=max(feasible,key=lambda r:(r['worst_view_gain'],r['net_q0']+r['net_a'],-r['radius'],r['name'])) if feasible else None
    lock=read(run/'source_selection_locked.json');chosen=choice['name'] if choice else 'baseline'
    assert lock['result']==choice and lock['selected']==chosen and lock['memory_texts_sha256']==core.digest(base)
    assert lock['selected_index_sha256']=={cid:sha(run/'indexes'/chosen/f'{cid}.npy') for cid in cids}
    assert not lock['selection_used_new_dev_outcomes'] and not lock['selection_used_audit_outcomes']
    assert training_lock['locked_at']<(run/'fit_baseline_source_view_fit_items.json').stat().st_mtime+1
    for name in vectors:assert lock['locked_at']<(run/f'{name}_dev_items.json').stat().st_mtime+1
    result={'source_hashes_verified':True,'optimizer_arrays_replayed':len(traces),'maximum_vector_replay_error':maximum_replay_error,
            'original_reference_rows_exact':70,'retrieval_contexts_and_predictions_verified':contexts,'source_training_queries':97,
            'stored_texts_unchanged':True,'unit_norm_and_radius_caps_verified':True,'dense_bytes_per_conversation':{cid:original[cid].nbytes for cid in cids},
            'all_selection_locks_verified':True,'source_selected':chosen,'optimizer_traces':traces}
    if rigid:
        result['cross_conversation_training_exclusion_verified']=True
        result['pairwise_document_geometry_preservation_verified']=True
    core.save(root/f'{args.out_prefix}_SOURCE.validation.json',result)
    history=read(run/'history.json');scores={r['name']:r['dev']['official_f1'] for r in history['rounds']};n=lambda c:sum(r['preserved'] for r in c.values())
    lines=['# 원문 감독 신호로 저장 벡터 최적화','',f"새 개발 평가 전 원문 질문 기준 선택은 **{chosen}**다. 개발 점수 기준 보관한 최고는 별도로 **{history['winner']} / {100*history['best_f1']:.2f}**다.",'',
           '| 조건/반경 | Q0 보존/75 | A 보존/75 | Q0 퇴행 | A 퇴행 | 개발 F1 |','|---|---:|---:|---:|---:|---:|',
           f"| 기준 | {n(contract['q0'])} | {n(contract['fit_a'])} | — | — | {100*p['reference']['expected_f1']:.2f} |"]
    for recipe in p['recipes']:
        name=recipe['name'];t=trials[name];r=t['result']
        lines.append(f"| {recipe.get('scope','')} {recipe['radius']} | {n(t['q0_contract'])} | {n(t['fit_a_contract'])} | {len(r['q0']['regressions'])} | {len(r['fit_a']['regressions'])} | {100*scores[name]:.2f} |")
    method='직교 정렬' if rigid else '200단계 학습'
    lines+=['',f"{len(traces)}개 벡터 배열의 {method}을 재현했고 최대 성분 오차는 {maximum_replay_error:.3g}였다. {contexts}개 실제 검색 문맥·출처·토큰·F1을 저장된 질문/문서 벡터로 재검증했다. 원래 평가기의 기준 예측 70개도 문맥까지 동일했다.",
            f"원문 저장량은 모든 후보에서 같고, float32 dense index는 대화 평균 {sum(v.nbytes for v in original.values())/len(cids):,.0f}바이트로 같다. 질의·문서 encoder와 reader 가중치를 바꾸지 않고 저장된 문서 벡터만 움직였다.",
            '학습은 source-fit Q0 97개만 사용했다. A 75개는 학습에 사용하지 않고 모든 후보 벡터를 고정한 뒤 선택에만 사용했다. source-ID 교집합은 약한 감독 신호이며 완전한 답 지지의 증명이 아니다. 두 표현의 순증가 기준은 관측된 개별 퇴행을 허용하므로 함께 보고한다.',
            '별도 audit100은 이미 소비한 고정 checkpoint이며 이 실험의 튜닝 입력이 아니다. 벡터의 작은 이동이나 source 학습 loss 감소가 새로운 질문의 F1 개선을 보장하지 않는다.','']
    if rigid:lines+=['전이 조건은 대상 대화의 source 질문을 전부 제외하고 다른 여섯 대화에서 정렬했다. 각 대화의 문서 간 내적이 보존되는 것과 모든 학습 질문의 대화 출처를 재검증했다. 이는 이미 반복 평가한 개발 대화의 새로운 일반화 점수는 아니다.','']
    (root/f'{args.out_prefix}_SOURCE_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='optimizer_traces'}))


if __name__=='__main__':main()
