"""FinRevBench scorer. Standard library only.

Usage: python score.py PREDICTIONS.jsonl --gold GOLD.jsonl [--parse strict|fence]
Predictions rows: {"question_id": ..., "raw_output": "<model text>"}; missing rows score as wrong.
--parse strict: the raw text must be one JSON object. --parse fence: one surrounding ```json fence is removed first.
"""
import argparse, json, re, sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

METRICS = ("answer_correct", "evidence_correct", "temporal_valid", "joint_correct")


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse(text, policy):
    text = (text or "").strip()
    if policy == "fence":
        m = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.S | re.I)
        if m:
            text = m.group(1)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def number(value):
    """One finite scalar at the requested scale; no rounding or unit guessing."""
    if isinstance(value, bool) or value is None:
        raise ValueError("Expected a finite numeric answer")
    text = str(value).strip().replace("−", "-")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()
    if "," in text:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
            raise ValueError("Malformed thousands separators")
        text = text.replace(",", "")
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Not a numeric scalar") from exc
    if not result.is_finite():
        raise ValueError("Answer must be finite")
    return result


def unit_signature(unit):
    text = re.sub(r"\s+", " ", str(unit).lower().replace(".", "")).strip()
    allowed = {"usd", "cad", "eur", "gbp", "jpy", "aud", "chf", "cny", "dollar", "dollars", "thousand", "thousands",
               "million", "millions", "in", "of", "as", "presented", "exact", "u", "s", "us", "united", "states", "canadian"}
    if re.search(r"\d", text) or set(re.findall(r"[a-z]+", text)) - allowed:
        raise ValueError("Unsupported unit")
    currencies = set(re.findall(r"\b(?:usd|cad|eur|gbp|jpy|aud|chf|cny)\b", text))
    if re.search(r"\b(?:us dollars|u s dollars|united states dollars)\b", text):
        currencies.add("usd")
    if "canadian dollars" in text:
        currencies.add("cad")
    thousand, million = bool(re.search(r"\bthousands?\b", text)), bool(re.search(r"\bmillions?\b", text))
    if len(currencies) > 1 or (thousand and million):
        raise ValueError("Ambiguous unit")
    if not currencies and not re.search(r"\b(?:dollars|thousands|millions)\b", text):
        raise ValueError("Unrecognized unit")
    return next(iter(currencies), None), 1_000_000 if million else 1000 if thousand else 1


def matching_unit(gold_unit, predicted_unit):
    try:
        gc, scale = unit_signature(gold_unit)
        pc, predicted_scale = unit_signature(predicted_unit)
    except ValueError:
        return False
    return scale == predicted_scale and (gc is None or pc == gc)  # "as presented" targets fix scale only


def same(expected, actual, field):
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if field == "status" or field.endswith("_status"):
        return actual == expected
    if expected is None:
        return actual is None
    try:
        return number(actual) == number(expected)
    except ValueError:
        return False


def score_hfkr(g, p):
    """Answer: every requested field exact. Evidence, per role: cited labels include an approved sufficient set
    and nothing outside the approved sources. Temporal: every cited source filed by the role's cutoff."""
    result = dict.fromkeys(METRICS, False)
    expected, actual = g["answer"], p.get("answer")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return result
    result["answer_correct"] = all(same(v, actual[k], k) for k, v in expected.items())
    requirements, evidence = g["evidence_requirements"], p.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(requirements):
        return result
    if g["operator"] == "supported_at_cutoff" and expected["value"] is None:
        if evidence["reported_value"] != []:  # a null supported value must cite no figure
            return result
        requirements = {"source_state": requirements["source_state"]}
    dates = g["source_dates"]
    evidence_ok = temporal_ok = True
    for role, req in requirements.items():
        cited = evidence[role]
        if (not isinstance(cited, list) or not cited or not all(isinstance(c, str) for c in cited)
                or len(cited) != len(set(cited)) or not all(c in dates for c in cited)):
            evidence_ok = temporal_ok = False
            continue
        groups = [set(s) for s in req["sufficient_sets"]]
        evidence_ok &= set(cited) <= set().union(*groups) and any(s <= set(cited) for s in groups)
        temporal_ok &= all(dates[c] <= req["cutoff"] for c in cited)
    result.update(evidence_correct=evidence_ok, temporal_valid=temporal_ok,
                  joint_correct=result["answer_correct"] and evidence_ok and temporal_ok)
    return result


def score_temporal(gold, pred):
    """Exact signed value at the requested display scale; pairs need both endpoints right."""
    res = {}
    for g in gold:
        p = pred.get(g["question_id"])
        try:
            if not p or "answer" not in p:
                raise ValueError
            value = number(p["answer"])
            ok = value == number(g["answer"]["answer"]) and (p.get("unit") is None or matching_unit(g["unit"], p["unit"]))
            res[g["question_id"]] = (ok, value)
        except ValueError:
            res[g["question_id"]] = (False, None)
    pairs, issuers = defaultdict(dict), defaultdict(list)
    for g in gold:
        pairs[g["case_id"]][g["side"]] = g
        issuers["|".join(g["issuer_group_id"])].append(res[g["question_id"]][0])
    both = delta = 0
    for pair in pairs.values():
        (pre_ok, pre), (post_ok, post) = res[pair["PRE"]["question_id"]], res[pair["POST"]["question_id"]]
        both += pre_ok and post_ok
        if pre is not None and post is not None:
            delta += post - pre == number(pair["POST"]["answer"]["answer"]) - number(pair["PRE"]["answer"]["answer"])
    side = lambda s: sum(res[g["question_id"]][0] for g in gold if g["side"] == s)
    return {"endpoints": len(gold), "endpoint_correct": sum(ok for ok, _ in res.values()), "pairs": len(pairs),
            "PRE_correct": side("PRE"), "POST_correct": side("POST"), "pair_correct": both,
            "delta_accuracy": delta / len(pairs),
            "issuer_macro_accuracy": sum(sum(v) / len(v) for v in issuers.values()) / len(issuers)}


def summarize(group):
    events = defaultdict(list)
    for d in group:
        events[d["event_id"]].append(d["joint_correct"])
    return {"questions": len(group), **{m: sum(d[m] for d in group) for m in METRICS},
            "event_macro": {"events": len(events),
                            "joint_correct": sum(sum(v) / len(v) for v in events.values()) / len(events)}}


def score(gold, raw, policy):
    pred = {q: parse(t, policy) for q, t in raw.items()}
    hfkr = defaultdict(list)
    for g in gold:
        if g["track"] != "temporal":
            hfkr[g["track"]].append({"event_id": g["event_id"], **score_hfkr(g, pred.get(g["question_id"]) or {})})
    out = {"parse": policy, "valid_json": sum(p is not None for p in pred.values()), "questions": len(gold),
           "hfkr": {t: summarize(v) for t, v in sorted(hfkr.items())}}
    out["hfkr"]["total"] = summarize([d for v in hfkr.values() for d in v])
    out["temporal"] = score_temporal([g for g in gold if g["track"] == "temporal"], pred)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--parse", choices=("strict", "fence"), default="strict")
    references = ap.add_mutually_exclusive_group(required=True)
    references.add_argument("--gold", type=Path, help="path to the private reference answers")
    references.add_argument("--data", type=Path, help="directory containing gold.jsonl")
    a = ap.parse_args()
    gold_path = a.gold if a.gold is not None else a.data / "gold.jsonl"
    if not gold_path.is_file():
        ap.error(f"reference answers not found at {gold_path}")
    gold = rows(gold_path)
    raw = {r["question_id"]: r["raw_output"] for r in rows(a.predictions)}
    unknown = set(raw) - {g["question_id"] for g in gold}
    if unknown:
        sys.exit(f"{len(unknown)} prediction IDs are not benchmark questions")
    print(json.dumps(score(gold, raw, a.parse), indent=2))
