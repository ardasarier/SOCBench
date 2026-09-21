"""
Step 3.3 of Algorithm 1: segment-level compression (Eq. 3).

Keeps the segments containing the top-k highest-attention tokens, plus the
endpoint's identity unconditionally.

DEVIATION FROM THE PAPER. Eq. 3 retains whole SENTENCES, which assumes prose.
socrag chunks are `Endpoint: VERB /path\\nSpecification:\\n{minified JSON}`, and
sentence splitting on that is not merely imprecise, it is broken:

  - The regex requires punctuation followed by whitespace. In minified JSON a
    period is followed by `"`, so `"...on TMDb.", "responses"` never matches.
  - Measured on GET /movie/top_rated (14,378 chars): the first sentence span
    began at char 584. Everything before it -- the endpoint path, the summary,
    and the endpoint's own description -- sat in a span the splitter never
    emitted, so it could not be retained at any k.
  - What survived instead were movie plot summaries from the response example
    payloads.

Compression was stripping an endpoint's identity and keeping its example data.
That explains the failure mode measured on TMDB, where the composition model
picked plausible neighbours (/movie/top_rated -> /movie/popular,
/movie/{id}/release_dates -> /discover/movie) and recall fell ~0.10.

This module segments on JSON structure instead, and retains the endpoint's
identity by KEY rather than by a character prefix. A fixed prefix was tried
first and does not work across benchmarks: SOCBench-D's top-level descriptions
range from 142 to 370 chars, so a prefix long enough for all of them (350)
made 52.7% of the text incompressible, capping the achievable ratio at ~1.9x.
Matching on keys drops that floor to 27%.

Sentence splitting remains as a fallback for non-JSON text.

Pure functions -- no model imports at module level, so importing this is cheap.

Demo: python -m attentionrag.compression   (from src/)
"""
import random
import re
import zlib

# Chunks produced by socrag's OpenApiParser start with this header.
HEADER_PATTERN = re.compile(r"\A(Endpoint:[^\n]*\n(?:Specification:\n)?)")

MAX_SEGMENT_CHARS = 120         # recurse into anything larger
MIN_SEGMENT_CHARS = 25          # below this, a split produces scraps not content
MAX_SEGMENT_DEPTH = 20          # guard against pathologically nested specs

# What makes an endpoint distinguishable from its siblings. Retained whole and
# unconditionally -- see the module docstring for why a character prefix fails.
IDENTITY_KEYS = ('"summary"', '"description"', '"operationId"')


def _top_level_spans(text: str, offset: int):
    """Char spans of each comma-separated item at brace depth 1, ignoring
    punctuation inside string literals."""
    spans = []
    depth = 0
    in_string = False
    escaped = False
    segment_start = None

    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
            if depth == 1:
                segment_start = i + 1
        elif char in "}]":
            if depth == 1 and segment_start is not None:
                spans.append((offset + segment_start, offset + i))
                segment_start = None
            depth -= 1
        elif char == "," and depth == 1 and segment_start is not None:
            spans.append((offset + segment_start, offset + i))
            segment_start = i + 1

    return spans


def _segment_json(text: str, start: int, end: int, out: list, depth: int = 0) -> None:
    # Identity fields are kept whole regardless of length -- splitting a
    # description would leave only its first fragment matchable by key.
    if text[start:end].lstrip().startswith(IDENTITY_KEYS):
        if text[start:end].strip():
            out.append((text[start:end].strip(), start, end, True))
        return

    if end - start <= MAX_SEGMENT_CHARS or depth >= MAX_SEGMENT_DEPTH:
        if text[start:end].strip():
            out.append((text[start:end].strip(), start, end, False))
        return

    spans = _top_level_spans(text[start:end], start)

    if len(spans) == 1 and spans[0][1] - spans[0][0] >= end - start - 2:
        _segment_json(text, spans[0][0], spans[0][1], out, depth + 1)
        return

    # No balanced structure found -- this is a mid-JSON continuation fragment
    # from NaiveTextParser (endpoints over 1024 tokens get split, and only the
    # first fragment carries the header). Emitting it whole would retain 4000
    # chars on a single token hit, so fall back to fixed-width windows.
    if not spans:
        for window_start in range(start, end, MAX_SEGMENT_CHARS):
            window_end = min(window_start + MAX_SEGMENT_CHARS, end)
            if text[window_start:window_end].strip():
                out.append((text[window_start:window_end].strip(),
                            window_start, window_end, False))
        return

    # Arrays of bare scalars (e.g. "genre_ids": [18, 10402]) split into individual
    # numbers. Retaining "10402" as a whole chunk is worse than useless -- it wastes
    # a top-k slot. If every span would be a scrap, keep the parent intact.
    if max(e - s for s, e in spans) < MIN_SEGMENT_CHARS:
        if text[start:end].strip():
            out.append((text[start:end].strip(), start, end, False))
        return

    for span_start, span_end in spans:
        _segment_json(text, span_start, span_end, out, depth + 1)


def _split_sentences(body: str, offset: int, out: list) -> None:
    for match in re.finditer(r"[^.!?]*[.!?]+(?:\s+|$)", body):
        if body[match.start():match.end()].strip():
            out.append((
                body[match.start():match.end()].strip(),
                offset + match.start(),
                offset + match.end(),
                False,
            ))


def split_into_segments(text: str):
    """Returns (segment, char_start, char_end, always_keep) tuples."""
    segments = []
    header = HEADER_PATTERN.match(text)
    body_start = 0

    if header:
        # The `Endpoint: VERB /path` line is the chunk's identity and is always
        # kept. The identity FIELDS are marked by key in _segment_json.
        segments.append((header.group(1).strip(), 0, header.end(), True))
        body_start = header.end()

    body = text[body_start:]
    if "{" in body or "[" in body:
        _segment_json(text, body_start, len(text), segments)
    else:
        _split_sentences(body, body_start, segments)

    # _segment_json marks every identity-keyed segment, but "responses" and
    # "parameters" carry their own nested "description" fields. Only the FIRST
    # occurrence of each key is the endpoint's own identity. Keeping the rest
    # unconditionally raised the incompressible floor from 27% to 42% on
    # SOCBench-D, and would take the selection job away from the attention
    # scores -- those nested descriptions are exactly what the method should
    # be choosing between.
    seen = set()
    result = []
    for segment, start, end, always_keep in segments:
        if always_keep:
            key = next(
                (k for k in IDENTITY_KEYS if segment.lstrip().startswith(k)), None
            )
            if key is not None:
                if key in seen:
                    always_keep = False
                else:
                    seen.add(key)
        result.append((segment, start, end, always_keep))

    return result


def select_tokens(token_scores: list, k: int = None, threshold_ratio: float = None):
    """
    Two selection strategies.

    k                -- Eq. 3 as published: the k highest-scoring tokens.
    threshold_ratio  -- keep every token scoring at least
                        threshold_ratio * max_score for this chunk.

    The threshold is RELATIVE, not absolute, because raw attention scores are
    not comparable across chunks: they depend on sequence length and on how
    many layers were summed.

    It also adapts to how peaked a chunk's distribution is, which fixed k
    cannot. Measured on a Spotify endpoint: top token 0.65, everything else
    clustered 0.16-0.25. k=5 retained four irrelevant parameter definitions;
    threshold_ratio=0.5 retains only the relevant one.
    """
    if not token_scores:
        return []
    if threshold_ratio is not None:
        cutoff = max(t["score"] for t in token_scores) * threshold_ratio
        return [t for t in token_scores if t["score"] >= cutoff]
    return sorted(token_scores, key=lambda t: -t["score"])[:k]


def compress_chunk(chunk: str, token_scores: list, k: int = 3,
                   threshold_ratio: float = None, identity_only: bool = False,
                   random_scores: bool = False, keep_identity: bool = True):
    """Returns (compressed_text, selected_entries)."""
    # Ablation: keep ONLY the unconditionally retained identity (path, summary,
    # description) and drop everything else. Tests whether the composition
    # model needs the endpoint's body at all, or whether it reconstructs
    # behaviour from the identity plus prior knowledge of the API.
    if identity_only:
        segments = split_into_segments(chunk)
        kept = [seg for seg, _, _, always_keep in segments if always_keep]
        # Continuation fragments carry no Endpoint: header, so nothing is
        # marked always_keep. Falling back to the first segment preserves the
        # chunk count, so the ablation isolates "identity only" rather than
        # also removing 86% of chunks.
        if not kept and segments:
            kept = [segments[0][0]]
        return " ".join(kept), []

    # Control condition: replace attention scores with random ones, keeping
    # token positions, segmentation and identity retention identical. If this
    # scores as well as real attention, the attention values carry no usable
    # signal and the attention-vs-identity comparison would really be
    # random-vs-identity. Seeded per chunk so reruns are reproducible.
    if random_scores:
        rng = random.Random(zlib.crc32(chunk.encode("utf-8")))
        token_scores = [{**t, "score": rng.random()} for t in token_scores]

    selected = select_tokens(token_scores, k=k, threshold_ratio=threshold_ratio)
    selected_starts = [t["char_start"] for t in selected]

    # keep_identity=False removes the unconditional retention entirely, so the
    # selection alone decides what survives -- including the Endpoint header.
    # Tests whether attention finds the endpoint's identity unaided, which the
    # always-keep rule otherwise does for it (and for random selection too).
    kept = [
        segment
        for segment, start, end, always_keep in split_into_segments(chunk)
        if (keep_identity and always_keep)
        or any(start <= token_start < end for token_start in selected_starts)
    ]
    return " ".join(kept), selected


if __name__ == "__main__":
    from .anchor_token import generate_anchor_token
    from .attention_features import compute_attention_feature
    from .hint_prefix import generate_answer_hint_prefix

    question = "Where did the cat sit?"
    hint = generate_answer_hint_prefix(question)

    # Two sentences, so Eq. 3 has to actually choose one.
    chunk = (
        "The cat sat on the warm windowsill in the afternoon sun. "
        "Later, the dog barked loudly at a passing car."
    )

    anchor = generate_anchor_token(chunk, question, hint)
    print(f"Question:     {question}")
    print(f"Hint prefix:  {hint!r}")
    print(f"Anchor token: {anchor!r}\n")

    if anchor.strip().lower() == "none":
        print("skipped -- anchor token is 'none'")
    else:
        token_scores, num_layers = compute_attention_feature(chunk, question, hint, anchor)
        compressed, top_k = compress_chunk(chunk, token_scores, k=3)

        print(f"Top-{len(top_k)} tokens (summed over {num_layers} layers):")
        for entry in top_k:
            print(f"  {entry['token']!r:<15} {entry['score']:.4f}")

        print(f"\nOriginal   ({len(chunk)} chars): {chunk!r}")
        print(f"Compressed ({len(compressed)} chars): {compressed!r}")
