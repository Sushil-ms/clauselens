"""
ClauseLens — Day 1 drill: get to know your data before you build anything.

Run:   python day01_explore_cuad.py data/CUADv1.json
Goal:  every check prints PASS. Fill in each TODO, re-run, repeat.

Rules of the drill
- Standard library only today (json, re, statistics, hashlib, collections).
- Don't peek at solutions online. If you're stuck for 15 min, ask Claude for a HINT, not the answer.
- Each function raises NotImplementedError until you write it; the harness shows that as TODO.

CUAD file shape (SQuAD format):
{
  "version": "...",
  "data": [                                   # one entry per contract (510 total)
    {
      "title": "CompanyX_..._Agreement",
      "paragraphs": [                         # CUAD has exactly ONE paragraph per contract
        {
          "context": "<full contract text>",
          "qas": [                            # one question per clause category (41 total)
            {
              "id": "<title>__Governing Law",
              "question": "Highlight the parts (if any) of this contract related to \"Governing Law\" ...",
              "answers": [{"text": "...", "answer_start": 1234}, ...],   # [] when the clause is absent
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from typing import TypedDict


class QA(TypedDict):
    id: str
    question: str
    answers: list[dict]


class Contract(TypedDict):
    title: str
    text: str
    qas: list[QA]


# ---------------------------------------------------------------------------
# Drill 1 — Load the file into a flat, typed shape
# ---------------------------------------------------------------------------
def load_contracts(path: str) -> list[Contract]:
    """Return one Contract per entry in data["data"].

    text = the paragraph's "context"; qas = the paragraph's "qas".
    HINT: CUAD has one paragraph per contract, but write it so it would still
    work (by concatenating) if there were several. Why? Real legal data is messy.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Drill 2 — Which clause category is a question about?
# ---------------------------------------------------------------------------
def category_of(qa: QA) -> str:
    """Return the clause category, e.g. "Governing Law".

    Two sources of truth: the id ends with "__<Category>", and the question
    contains the category in double quotes. Use the id first, the question as a
    fallback (regex). Strip whitespace.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Drill 3 — Coverage: how often does each clause actually appear?
# ---------------------------------------------------------------------------
def clause_coverage(contracts: list[Contract]) -> dict[str, float]:
    """Return {category: fraction of contracts with at least one answer}.

    Values between 0.0 and 1.0. This tells you which clauses are rare — and rare
    clauses are where your RAG system and your fine-tuned model will struggle.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Drill 4 — Data integrity: do the answer spans really point into the text?
# ---------------------------------------------------------------------------
def span_match_rate(contracts: list[Contract]) -> float:
    """Return the fraction of answers where
    text[answer_start : answer_start + len(answer_text)] == answer_text.

    Backend habit: never trust a dataset's offsets until you've checked them.
    Later, your evals will compare retrieved chunks against these spans.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Drill 5 — How long are contracts? (This decides your chunking strategy.)
# ---------------------------------------------------------------------------
def length_stats(contracts: list[Contract]) -> dict[str, int]:
    """Return approximate token stats using tokens ≈ len(text) // 4.

    Keys: "p50", "p95", "max" (all ints).
    HINT: statistics.quantiles(values, n=100) gives 99 cut points; index 49 is
    p50, index 94 is p95.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Drill 6 — A leak-free, reproducible train/test split
# ---------------------------------------------------------------------------
def split_by_contract(
    contracts: list[Contract], test_fraction: float = 0.2
) -> tuple[list[Contract], list[Contract]]:
    """Split by CONTRACT (never by question), deterministically.

    Put a contract in test if int(sha256(title).hexdigest(), 16) % 100 < test_fraction * 100.
    Why by contract? If clauses from the same contract land in train and test,
    your fine-tuned model "remembers" the contract and your eval numbers lie.
    Why a hash, not random.shuffle? Same split on every machine, every run, forever.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Harness — don't edit below this line
# ---------------------------------------------------------------------------
def _run(name, fn):
    try:
        ok, detail = fn()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return "PASS" if ok else "FAIL"
    except NotImplementedError:
        print(f"[TODO] {name}")
        return "TODO"
    except Exception as e:  # noqa: BLE001 — show the learner what broke
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return "FAIL"


def main(path: str) -> None:
    state: dict = {}

    def d1():
        cs = load_contracts(path)
        state["cs"] = cs
        ok = (
            len(cs) == 510
            and all(c["text"] and c["title"] for c in cs)
            and all(len(c["qas"]) == 41 for c in cs)
        )
        return ok, f"{len(cs)} contracts, {len(cs[0]['qas']) if cs else 0} questions each (want 510 / 41)"

    def d2():
        cs = state["cs"]
        cats = {category_of(q) for c in cs for q in c["qas"]}
        state["cats"] = cats
        fake = {"id": "weird_title_without_suffix", "question": 'Highlight ... related to "Governing Law" that ...', "answers": []}
        fallback_ok = category_of(fake) == "Governing Law"
        ok = len(cats) == 41 and "Governing Law" in cats and fallback_ok
        return ok, f"{len(cats)} categories (want 41), regex fallback {'works' if fallback_ok else 'BROKEN'}"

    def d3():
        cov = clause_coverage(state["cs"])
        ok = len(cov) == 41 and all(0.0 <= v <= 1.0 for v in cov.values())
        common = sorted(cov.items(), key=lambda kv: -kv[1])[:3]
        rare = sorted(cov.items(), key=lambda kv: kv[1])[:3]
        fmt = lambda xs: ", ".join(f"{k} {v:.0%}" for k, v in xs)  # noqa: E731
        return ok, f"most common: {fmt(common)} | rarest: {fmt(rare)}"

    def d4():
        rate = span_match_rate(state["cs"])
        return rate >= 0.99, f"{rate:.2%} of answer spans match the text (want ≥ 99%)"

    def d5():
        s = length_stats(state["cs"])
        ok = set(s) == {"p50", "p95", "max"} and s["p50"] <= s["p95"] <= s["max"]
        return ok, f"~tokens per contract: p50 {s['p50']:,}, p95 {s['p95']:,}, max {s['max']:,}"

    def d6():
        cs = state["cs"]
        tr, te = split_by_contract(cs)
        tr2, te2 = split_by_contract(cs)
        overlap = {c["title"] for c in tr} & {c["title"] for c in te}
        stable = [c["title"] for c in te] == [c["title"] for c in te2]
        frac = len(te) / len(cs)
        ok = not overlap and stable and len(tr) + len(te) == len(cs) and 0.12 <= frac <= 0.28
        return ok, f"train {len(tr)} / test {len(te)} ({frac:.0%}), overlap {len(overlap)}, deterministic {stable}"

    results = []
    for name, fn in [
        ("Drill 1  load_contracts", d1),
        ("Drill 2  category_of", d2),
        ("Drill 3  clause_coverage", d3),
        ("Drill 4  span_match_rate", d4),
        ("Drill 5  length_stats", d5),
        ("Drill 6  split_by_contract", d6),
    ]:
        if name != "Drill 1  load_contracts" and "cs" not in state:
            print(f"[SKIP] {name}: finish Drill 1 first")
            results.append("SKIP")
            continue
        results.append(_run(name, fn))

    c = Counter(results)
    print(f"\n{c['PASS']}/6 PASS · {c['FAIL']} FAIL · {c['TODO']} TODO · {c['SKIP']} SKIP")
    if c["PASS"] == 6:
        print("Day 1 done. Paste the output to Claude: the numbers decide Day 2's chunking design.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/CUADv1.json")
