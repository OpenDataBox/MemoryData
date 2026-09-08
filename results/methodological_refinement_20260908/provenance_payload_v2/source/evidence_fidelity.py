"""Source-only behavior fidelity checks for measured evidence proposals."""
import re
from evidence_utility import POLICY,context_for

VERIFIER = (
    'Assess whether a compressed conversation fragment preserves an answer. '
    'The ORIGINAL source is authoritative. The question and answers are source-generated diagnostics. '
    'Output EQUIVALENT only if the candidate answer means the same thing as the reference answer, '
    'with the same person, object, event, time, quantity, polarity, and conditions, AND the CANDIDATE '
    'fragment by itself supports that answer to this question. Pronoun shifts in a faithful paraphrase '
    'and harmless wording differences are allowed. Missing identifying context, a changed relation, '
    'an unsupported guess, or an abstention instead of a supported answer is DIFFERENT. '
    'Use UNKNOWN if the original source does not support the reference or you cannot decide. '
    'First line must be exactly EQUIVALENT, DIFFERENT, or UNKNOWN. No other text.'
)


def accepted(text):
    return bool(re.fullmatch(r'EQUIVALENT[.!]?',text.strip().upper()))


def verify(rt,sessions_by_id,rows):
    pending=[]
    for row in sorted(rows,key=lambda r:r['id']):
        if row['split']!='probe_fit':
            raise ValueError('Audit cannot enter source-fidelity refinement')
        if 'analysis' not in row or not row['analysis']['source_dependent'] or row['generations']['full']['source_answer_f1']<POLICY['minimum_full_answer_f1']:
            continue
        turns={t['id']:t for s in sessions_by_id[row['conv_id']] for t in s['turns']}
        full=next(s for s in row['subsets'] if s['kind']=='full')
        original=context_for(row,turns,full['source_ids'])
        reference=row['generations']['full']['text']
        seen=set()
        for name in ('best_single','best_pair','minimum_bundle','single_policy'):
            candidate=row['generations'][name]
            ids=tuple(candidate['source_ids'])
            if ids in seen or list(ids)==full['source_ids']:
                continue
            seen.add(ids)
            fragment=context_for(row,turns,ids)
            prompt=(f"ORIGINAL source:\n{original}\n\nQuestion: {row['question']}\nReference answer: {reference}"
                    f"\n\nCANDIDATE fragment:\n{fragment}\nCandidate answer: {candidate['text']}\n\nVerdict:")
            pending.append((row['id'],'|'.join(ids),prompt))
    outputs=rt.generate(VERIFIER,[p[2] for p in pending],max_tokens=16)
    result={}
    for (probe_id,key,prompt),text in zip(pending,outputs):
        result.setdefault(probe_id,{})[key]={'accepted':accepted(text),'verdict':text}
    return result
