"""CPU-only smoke test of data loading and probe parsing (no GPU needed)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from certmem.locomo import load, token_f1
from certmem.probes import parse

convs = load(sys.argv[1] if len(sys.argv) > 1 else "data/locomo10.json")
cid, c = next(iter(convs.items()))
print("convs:", len(convs), "| first:", cid, "| sessions:", len(c["sessions"]), "| qa(non-adv):", len(c["qa"]))
s = c["sessions"][0]; print("session 1 date:", s.date, "| turns:", len(s.turns))
print("sample turn:", s.turns[0].speaker, "-", s.turns[0].text[:80])
q = c["qa"][0]; print("sample qa:", q.question, "->", q.answer, "| evidence sessions:", q.sessions(), "| cat:", q.category)
covered = sum(1 for q in c["qa"] if q.sessions())
print(f"qa with evidence->session mapping: {covered}/{len(c['qa'])}")
print("token_f1 check:", token_f1("Becoming Nicole", "Becoming Nicole"), token_f1("the book", "Becoming Nicole"))
fake = '[{"question":"When?","answer":"' + s.turns[0].text.split()[0] + '","family":"date"}]'
print("probe parse:", parse(fake, s.text()))
