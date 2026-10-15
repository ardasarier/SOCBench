"""
Which part of an endpoint's body does a selection strategy keep?

At matched budget, random selection set more required body fields than
attention (0.539 vs 0.423). Hypothesis: attention prefers text that is
semantically close to the query -- response descriptions like "Successfully
retrieved operational status ..." -- over the request schema (parameters,
requestBody) that is needed to build a call correctly.

This measures, per condition, how the kept non-identity text splits across the
top-level fields of the operation: request (parameters, requestBody),
response (responses) and other. Free, no models: kept segments are recovered
from the saved contexts, since compress_chunk joins them in order with " ".

Run from socbench-d/code/:
    python src/body_selection.py
"""
import json
import re
from collections import Counter

from attentionrag.compression import HEADER_PATTERN, _top_level_spans, split_into_segments

C = "data/contexts/socbenchd_1_k10_all_noanchor_"
CONDITIONS = {
    "attention+rule": C + "k10_n110.json",
    "random+rule": C + "random_n110.json",
}
REGION = {
    "summary": "identity", "description": "identity", "operationId": "identity",
    "parameters": "request", "requestBody": "request",
    "responses": "response",
}


def regions(chunk: str):
    """[(start, end, region)] of the operation's top-level fields."""
    header = HEADER_PATTERN.match(chunk)
    if not header:
        return None, None
    out = []
    for s, e in _top_level_spans(chunk[header.end():], header.end()):
        m = re.match(r'\s*"([^"]+)"\s*:', chunk[s:e])
        out.append((s, e, REGION.get(m.group(1), "other") if m else "other"))
    return header.end(), out


def region_of(pos: int, regs) -> str:
    for s, e, r in regs:
        if s <= pos < e:
            return r
    return "other"


def kept_flags(raw: str, compressed: str):
    """Segments of raw, each with a flag whether it survived. Kept segments
    appear in compressed in order, separated by single spaces."""
    p = 0
    for seg, s, e, _ in split_into_segments(raw):
        if compressed.startswith(seg, p):
            yield seg, s, True
            p += len(seg) + 1
        else:
            yield seg, s, False


def measure(path: str):
    raw_chars, kept_chars = Counter(), Counter()
    for rec in json.load(open(path)):
        if len(rec["raw_chunks"]) != len(rec["compressed_chunks"]):
            continue                      # dropped chunks break the alignment
        for raw, comp in zip(rec["raw_chunks"], rec["compressed_chunks"]):
            header_end, regs = regions(raw)
            if regs is None:
                continue
            for seg, start, kept in kept_flags(raw, comp):
                if start < header_end:
                    continue
                region = region_of(start, regs)
                if region == "identity":
                    continue
                raw_chars[region] += len(seg)
                if kept:
                    kept_chars[region] += len(seg)
    return raw_chars, kept_chars


def shares(counter: Counter) -> str:
    total = sum(counter.values()) or 1
    return "".join(f"{100 * counter[r] / total:>10.1f}%" for r in ("request", "response", "other"))


print("Share of NON-identity body text, by top-level field\n")
print(f"{'':<22}{'request':>11}{'response':>11}{'other':>11}")
reference = None
for name, path in CONDITIONS.items():
    raw, kept = measure(path)
    if reference is None:
        reference = raw
        print(f"{'available (raw)':<22}{shares(raw)}")
    print(f"{name:<22}{shares(kept)}")
