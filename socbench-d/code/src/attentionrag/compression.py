"""
Step 3.3 of Algorithm 1: segment-level compression (Eq. 3).

Keeps the segments containing the top-k highest-attention tokens.

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

This module segments on JSON structure instead, and always retains the
`Endpoint:` header since it is the chunk's identity. Sentence splitting remains
as a fallback for non-JSON text.

Pure functions -- no model imports at module level, so importing this is cheap.

Demo: python -m attentionrag.compression   (from src/)
"""
import re

# Chunks produced by socrag's OpenApiParser start with this header.
HEADER_PATTERN = re.compile(r"\A(Endpoint:[^\n]*\n(?:Specification:\n)?)")

MAX_SEGMENT_CHARS = 120  # recurse into anything larger
MIN_SEGMENT_CHARS = 25   # below this, a split produces scraps not content
MAX_SEGMENT_DEPTH = 20   # guard against pathologically nested specs

IDENTITY_PREFIX_CHARS = 350  # operationId + summary + description always lead the JSON


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
        # The endpoint's identity -- path, operationId, summary, description --
        # is what distinguishes it from its siblings (/movie/top_rated vs
        # /movie/popular). Leaving it to attention meant losing it whenever the
        # top-k tokens landed in the response example payloads, which is the
        # failure this whole module exists to fix. socrag serialises these three
        # keys first, in order, so a positional prefix captures them reliably.
        identity_end = min(header.end() + IDENTITY_PREFIX_CHARS, len(text))
        segments.append((text[:identity_end].strip(), 0, identity_end, True))
        body_start = identity_end

    body = text[body_start:]
    if "{" in body or "[" in body:
        _segment_json(text, body_start, len(text), segments)
    else:
        _split_sentences(body, body_start, segments)

    return segments


def compress_chunk(chunk: str, token_scores: list, k: int = 3):
    """Returns (compressed_text, top_k_entries)."""
    top_k = sorted(token_scores, key=lambda t: -t["score"])[:k]
    top_k_starts = [t["char_start"] for t in top_k]

    kept = [
        segment
        for segment, start, end, always_keep in split_into_segments(chunk)
        if always_keep or any(start <= token_start < end for token_start in top_k_starts)
    ]
    return " ".join(kept), top_k


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
