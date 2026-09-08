"""Compare the pilot scorer to only the F1 functions extracted from official code.

AST extraction avoids importing unrelated BERTScore dependencies or executing
top-level code from the downloaded reference.
"""
import ast
from collections import Counter
from pathlib import Path
import random
import string
import sys

import numpy as np
import regex
from nltk.stem import PorterStemmer

from refine import f1


def main(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    needed = {'normalize_answer', 'f1_score', 'f1'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in needed]
    if {node.name for node in selected} != needed:
        raise ValueError('Official reference layout changed')
    scope = {'regex':regex, 'string':string, 'Counter':Counter, 'np':np, 'ps':PorterStemmer()}
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), scope)
    rng = random.Random(123)
    vocab = ['Paris','London','dogs','dog','one','1','the','and','a','museum', 'art-school', 'May', '2023', '']
    cases = [('Paris, London','London, Paris'), ('dog','dogs'), ('May 7, 2023','7 May 2023'), ('','')]
    for _ in range(200):
        cases.append((' '.join(rng.choices(vocab,k=rng.randrange(9))),
                      ' '.join(rng.choices(vocab,k=rng.randrange(9)))))
    for pred, gold in cases:
        for category in (1,2,3,4):
            reference = gold.split(';')[0].strip() if category == 3 else gold
            expected = scope['f1' if category == 1 else 'f1_score'](pred, reference)
            observed = f1(pred,gold,category)
            assert abs(expected-observed) < 1e-12, (pred,gold,category,expected,observed)
    print(f'PASS: {len(cases)*4} comparisons against official LoCoMo F1')


if __name__ == '__main__':
    main(sys.argv[1])
