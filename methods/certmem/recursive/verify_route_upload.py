"""Record the exact newly authored code-only payload for approval review."""
import ast
import hashlib
import json
import re
from pathlib import Path


def main():
    folder=Path(__file__).parent
    names=['coverage_routes.py','test_coverage_routes.py','run_route_refinement.py','run_routes_4080s.sh','routes-supervisor.conf']
    files=[]
    for name in names:
        path=folder/name;data=path.read_bytes();text=data.decode('utf-8')
        if path.suffix=='.py':ast.parse(text)
        assert not any(marker in text for marker in ('PRIVATE KEY','sk-proj-','AKIA','@gmail.com'))
        assert not re.search(r'\bhf_[A-Za-z0-9]{30,}\b',text)
        files.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    result={'files':files,'total_bytes':sum(f['bytes'] for f in files),
            'payload_review':'All five complete files were inspected. They are newly authored experiment algorithm/runner/test/launch configuration from this task. The test contains synthetic source body / yes / no fixtures. No corpus, document, QA result, account credential or SSH key is in the upload.',
            'destination':'root@175.155.64.231:19679 /workspace/locomo-refinement/',
            'destination_evidence':'Read-only SSH with StrictHostKeyChecking=yes succeeded; it matched the already used rented RTX 4080 SUPER (32760 MiB), frozen evaluator and public LoCoMo corpus hashes.',
            'frozen_evaluator_sha256':'71384cac6aaa260d6df4f24f605797a8a886cd51514a83c9c1d041eff99f14fa',
            'public_locomo_sha256':'79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4',
            'public_dataset_source':'https://github.com/snap-research/locomo',
            'scope':'Continue the user-authorized refinement on the same rented 4080S; no new service or recipient and no extra GPU rental.'}
    target=Path('results/methodological_refinement_20260908/ROUTE_UPLOAD_PROVENANCE.json')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
