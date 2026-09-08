"""Audit the disjoint original/rebuilt Mem0 conversations and all incurred work."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_workers(source, indices):
    status = json.loads((source / 'parallel_status.json').read_text())
    if status.get('active_contexts'):
        raise ValueError('A source still reports active contexts')
    for index in indices:
        endings = [e for e in status['events'] if e['context'] == index and e['event'] == 'end']
        if len(endings) != 1 or endings[0]['exit_code'] != 0:
            raise ValueError(f'Context {index} lacks a unique successful process exit')
    return status


def repair_audit(source, indices, hashes):
    groups = defaultdict(list)
    for index in indices:
        path = source / f'context_{index:02d}' / 'mem0_action_repairs.jsonl'
        if not path.exists():
            continue
        hashes[str(path)] = digest(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups[(index, row['group'])].append(row)
    for group, rows in groups.items():
        if ([row['attempt'] for row in rows] != list(range(len(rows))) or not 2 <= len(rows) <= 3
                or rows[-1].get('accepted') is not True
                or any(row.get('accepted') is not False for row in rows[:-1])):
            raise ValueError(f'Unfinished or inconsistent semantic repair: {group}')
    return {'complete': True, 'repaired_plans': len(groups),
            'rejected_plans': sum(len(rows) - 1 for rows in groups.values()),
            'note': 'Invalid plans were rejected before memory mutation. All corrective model calls remain metered.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=Path('/workspace/MemoryData-finish-r5'))
    parser.add_argument('--plan', type=Path, default=Path('/workspace/finish-mem0-repair-plan-r5.json'))
    parser.add_argument('--prior-plan', type=Path, default=Path('/workspace/finish-mem0-plan-r3.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import yaml
    from scripts import locomo_server_queue as queue
    from scripts.finalize_langmem_recovery import shard_records
    from scripts.finalize_locomo_comparison import normalize_records, get_template
    from scripts.response_delivery_audit import audit_delivery
    from scripts.score_locomo_comparison import expected_questions, read_jsonl, write_report, build_report, summarize_usage, runtime_metrics

    plan, prior = (json.loads(path.read_text()) for path in (args.plan, args.prior_plan))
    queue.validate_plan(plan)
    hashes = {str(path): digest(path) for path in (args.plan, args.prior_plan, Path(__file__).resolve())}
    for path, checksum in prior['file_sha256'].items():
        if digest(path) != checksum:
            raise ValueError('Prior pinned file changed: ' + path)
    if plan['file_sha256'].get(str(args.prior_plan.resolve())) != digest(args.prior_plan):
        raise ValueError('Prior plan is not pinned by the repair attempt')
    for key in ('model', 'model_revision', 'dtype', 'embedding_path', 'dataset', 'scorer'):
        if plan.get(key) != prior.get(key):
            raise ValueError('Incompatible execution conditions: ' + key)
    prior_root = Path(next(item['agent_config'] for item in prior['methods'] if item['method'] == 'mem0')).parent.parent
    allowed_changes = {'methods/a_mem/source/a_mem/memory_layer.py',
        'methods/mem0/source/mem0/memory/main.py', 'scripts/locomo_server_queue.py',
        'utils/request_metering.py', 'scripts/finalize_locomo_comparison.py', 'scripts/run_remaining.py'}
    source_changes = []
    for path, checksum in prior['file_sha256'].items():
        old_file = Path(path)
        relative = old_file.relative_to(prior_root).as_posix() if old_file.is_relative_to(prior_root) else None
        new_file = str(args.source_root / relative) if relative else path
        if plan['file_sha256'].get(new_file) != checksum:
            if relative not in allowed_changes:
                raise ValueError('Undocumented source/config difference: ' + path)
            source_changes.append(relative)
    services = ('locomo-finish-mem0-r3', 'locomo-finish-mem0-r5')
    if args.wait:
        print(json.dumps({'state': 'waiting_for_generation', 'services': services}), flush=True)
        while any(queue.supervisor_state(service) in {'RUNNING', 'STARTING', 'STOPPING'} for service in services):
            time.sleep(30)
    for service in services:
        if queue.supervisor_state(service) != 'EXITED':
            raise ValueError('Required execution has not exited: ' + service)
    sources = [(Path(p['output']) / 'mem0').resolve() for p in (prior, plan)]
    # Include both proxy implementations; the shared vLLM server is not an artifact writer.
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = process.read_bytes().decode().split('\0')
        except FileNotFoundError:
            continue
        if (any(Path(arg).name in {'run_locomo_parallel.py', 'main.py', 'metered_openai_proxy.py', 'repairing_openai_proxy.py'} for arg in argv)
                and any(arg.startswith(str(source) + '/') for source in sources for arg in argv)):
            raise ValueError('Artifact writer remains live: ' + process.parent.name)
    destination = args.output.resolve()
    if destination.exists() or any(destination.is_relative_to(s) or s.is_relative_to(destination) for s in sources):
        raise ValueError('Composite destination must be new and separate from original attempts')
    selection_path = Path(plan['output']) / 'selection.json'
    selection = json.loads(selection_path.read_text())
    old_indices, new_indices = selection['preserved_contexts'], selection['rebuild_contexts']
    if (not new_indices or old_indices != sorted(set(old_indices)) or new_indices != sorted(set(new_indices))
            or set(old_indices) & set(new_indices) or sorted(old_indices + new_indices) != list(range(10))):
        raise ValueError('Source partitions are not a disjoint full ten-conversation set')
    old_status = terminal_workers(sources[0], old_indices)
    new_status = terminal_workers(sources[1], new_indices)
    actual_failed = sorted(e['context'] for e in old_status['events'] if e['event'] == 'end' and e['exit_code'] != 0)
    if actual_failed != new_indices:
        raise ValueError('Rebuilt contexts do not exactly match original failed workers')
    if new_status.get('selected_contexts_complete') is not True or new_status.get('context_indices') != new_indices or new_status.get('failed') is not False:
        raise ValueError('Rebuilt attempt did not complete its requested context set')
    attempt_path = sources[1] / 'selected_attempt.json'
    attempt = json.loads(attempt_path.read_text())
    if (attempt.get('run_id') != plan['run_id'] or attempt.get('method') != 'mem0'
            or attempt.get('context_indices') != new_indices or attempt.get('requires_composite') is not True):
        raise ValueError('Selected attempt metadata is inconsistent')
    expected = expected_questions(plan['dataset'], 1540)
    samples = list(dict.fromkeys(sample for sample, _ in expected))
    raw_groups = [shard_records(source, indices, expected, hashes)
                  for source, indices in zip(sources, (old_indices, new_indices))]
    selected_path = sources[1] / 'selected_results.json'
    if json.loads(selected_path.read_text())['data'] != raw_groups[1]:
        raise ValueError('Merged rebuilt results differ from their original worker artifacts')
    items = [next(item for item in p['methods'] if item['method'] == 'mem0') for p in (prior, plan)]
    for key in ('agent_config', 'dataset_config'):
        if digest(items[0][key]) != digest(items[1][key]):
            raise ValueError('Mem0 inference/dataset configuration differs between attempts')
    agent, config = (yaml.safe_load(Path(items[1][key]).read_text()) for key in ('agent_config', 'dataset_config'))
    template = get_template(config['sub_dataset'], 'query', agent['agent_name'])
    normalized = normalize_records(raw_groups[0] + raw_groups[1], expected, template)
    selected_usage, all_usage, execution = [], [], []
    for source, indices, p in zip(sources, (old_indices, new_indices), (prior, plan)):
        usage = read_jsonl(source / 'usage.jsonl')
        chosen_samples = {samples[i] for i in indices}
        chosen = [row for row in usage if row.get('sample_id') in chosen_samples]
        subset = {key: qa for key, qa in expected.items() if key[0] in chosen_samples}
        queue.audit_coverage(chosen, subset, p['run_id'], 'mem0')
        selected_usage.extend(chosen)
        all_usage.extend(usage)
        execution.append({'run_id': p['run_id'], 'output': str(source),
            'usage': summarize_usage(usage, True),
            'runtime': runtime_metrics([], source / 'telemetry.json', p.get('hourly_rate_usd'))})
        for path in (source / 'usage.jsonl', source / 'telemetry.json', source / 'parallel_status.json'):
            hashes[str(path)] = digest(path)
    delivery = audit_delivery(selected_usage)
    semantic = repair_audit(sources[1], new_indices, hashes)
    total_usage = summarize_usage(all_usage, True)
    if not delivery['complete'] or not total_usage['complete']:
        raise ValueError('Delivered response or total incurred accounting audit failed')
    for path in (selection_path, attempt_path, selected_path):
        hashes[str(path)] = digest(path)
    destination.mkdir(parents=True)
    normalized_path = destination / 'normalized_results.json'
    write_report(normalized_path, {'data': normalized})
    predictions = queue.canonical_predictions([normalized_path], expected)
    for name, rows in (('predictions.jsonl', predictions), ('selected_usage.jsonl', selected_usage), ('all_attempt_usage.jsonl', all_usage)):
        with (destination / name).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = build_report(plan['dataset'], destination / 'predictions.jsonl', plan['scorer'], destination / 'selected_usage.jsonl')
    report.update(method='mem0', run_ids=[prior['run_id'], plan['run_id']],
        response_delivery=delivery, semantic_plan_repairs=semantic, raw_sources_sha256=hashes,
        documented_source_changes=source_changes,
        selection=selection, all_attempt_usage=total_usage, execution_attempts=execution,
        failed_original_context_usage=summarize_usage([row for row in all_usage
            if row['run_id'] == prior['run_id'] and row['sample_id'] in {samples[i] for i in new_indices}], True),
        recovery_protocol=plan['recovery_protocol'],
        runtime_note='Runtime and rental estimates are reported for each entire attempt, including failed work. No partial-run GPU time is inferred by QA proportion.')
    report['complete'] = report['complete'] and report['empty_predictions'] == 0 and delivery['complete'] and semantic['complete'] and total_usage['complete']
    write_report(destination / 'report.json', report)
    if not report['complete']:
        raise ValueError('Composite failed full prediction/accounting completeness gates')
    print(json.dumps({key: report[key] for key in ('method', 'complete', 'evaluated', 'official_f1')}))


if __name__ == '__main__':
    main()
