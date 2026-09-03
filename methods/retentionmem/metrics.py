"""LoCoMo QA scoring protocol, separate from generic token F1.

Reference: snap-research/locomo/task_eval/evaluation.py, inspected 2026-09-03.
https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py
Includes Porter stemming, per-category reference handling and multi-answer matching.
"""
from collections import Counter
import re
import string
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()


def _words(value):
    stripped = str(value).lower().translate(str.maketrans("", "", string.punctuation))
    stripped = re.sub(r"\b(?:a|an|the|and)\b", " ", stripped)
    return [_stemmer.stem(word) for word in stripped.split()]


def _match(prediction, reference):
    pred, gold = _words(prediction), _words(reference)
    shared = sum((Counter(pred) & Counter(gold)).values())
    return (2.0*shared/(len(pred)+len(gold))) if shared else 0.0


def locomo_score(prediction, reference, category):
    category = int(category)
    if category == 1:
        predictions, answers = str(prediction).split(","), str(reference).split(",")
        return sum(max(_match(p, a) for p in predictions) for a in answers)/len(answers)
    if category in {2, 3, 4}:
        return _match(prediction, str(reference).split(";")[0].strip() if category == 3 else reference)
    if category == 5:
        return float(any(phrase in prediction.lower() for phrase in ("no information available", "not mentioned")))
    raise ValueError("Unknown LoCoMo category")
