"""Inspect the exact code-only identity-route payload and record its hashes."""
import ast
import hashlib
import json
from pathlib import Path
import re


def main():
    folder=Path(__file__).parent
    names=['identity_routes.py','test_identity_routes.py','run_identity_route_refinement.py',
           'run_identity_routes_4080s.sh','identity-routes-supervisor.conf','status_routes.py','snapshot_methodological.py']
    files=[]
    for name in names:
        path=folder/name;data=path.read_bytes();source=data.decode('utf-8')
        if path.suffix=='.py':ast.parse(source)
        assert not any(marker in source for marker in ('PRIVATE KEY','sk-proj-','AKIA','@gmail.com'))
        assert not re.search(r'\bhf_[A-Za-z0-9]{30,}\b',source)
        files.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    result={'files':files,'total_bytes':sum(f['bytes'] for f in files),
            'contents':'Newly authored public experiment code, synthetic toy tests, status/snapshot helpers and launch configuration. No actual corpus, benchmark QA, prediction, user document, account credential or private key is included.',
            'destination':'root@175.155.64.231:19679 /workspace/locomo-refinement/',
            'authorization':'Same user-rented RTX 4080S used for the ongoing LoCoMo refinement. StrictHostKeyChecking=yes verifies the existing host key; no new recipient, service or GPU rental.',
            'prior_host_evidence':'ROUTE_UPLOAD_PROVENANCE.json',
            'public_dataset_source':'https://github.com/snap-research/locomo'}
    target=Path('results/methodological_refinement_20260908/IDENTITY_ROUTES_UPLOAD_PROVENANCE.json')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
