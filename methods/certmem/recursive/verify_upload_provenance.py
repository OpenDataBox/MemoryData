"""Verify the proposed GPU payload derives solely from public/already-remote data."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import refine as core
from compile_research_data import cards_from_generations,split_probe_pool


def main():
    root=Path(__file__).resolve().parents[3]
    directory=root/'results/methodological_refinement_20260908'
    corpus=json.loads((directory/'source_sessions.json').read_text(encoding='utf-8'))
    pool=json.loads((directory/'source_probes.json').read_text(encoding='utf-8'))
    with gzip.open(directory/'source_qa_generations.json.gz','rt',encoding='utf-8') as stream:
        dump=json.load(stream)
    assert split_probe_pool(corpus['sessions_by_id'],cards_from_generations(dump))==pool
    samples=json.loads((root/'.runtime-cache/locomo-refinement/locomo10.json').read_text(encoding='utf-8'))
    assert corpus['dataset_sha256']==dump['dataset_sha256']==core.digest(samples)
    assert corpus['sessions_by_id']=={str(s['sample_id']):core.session_data(s) for s in samples
                                     if str(s['sample_id']) in corpus['sessions_by_id']}
    archive=root/'.runtime-cache/locomo-refinement/evidence_upload.tgz'
    with tarfile.open(archive) as stream:
        members=[m.name for m in stream if m.isfile()]
    expected={'./'+name for name in ('evidence_utility.py','run_evidence_utility.py','test_evidence_utility.py',
                'run_evidence_utility_4080s.sh','evidence-utility-supervisor.conf','transition_evidence_utility.py',
                'transition-evidence-supervisor.conf','research_data/source_sessions.json','research_data/source_probes.json')}
    assert set(members)==expected
    value={'public_source':'https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json',
           'public_source_sha256':hashlib.sha256((root/'.runtime-cache/locomo-refinement/locomo10.json').read_bytes()).hexdigest(),
           'already_remote_generation_dump_sha256':hashlib.sha256((directory/'source_qa_generations.json.gz').read_bytes()).hexdigest(),
           'source_sessions_exact_public_subset':True,'source_probes_exact_recompilation_of_remote_generations':True,
           'payload_contains':'Source conversation text and model-generated diagnostic questions/answers from public LoCoMo; no private user data or credentials; benchmark QA fields excluded.',
           'members':sorted(members),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    core.save(directory/'upload_provenance.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    main()
