"""
Step 3.3 of Algorithm 1: sentence-level compression (Eq. 3).

Keeps the sentences containing the top-k highest-attention tokens.

Pure functions -- no model imports at module level, so importing this costs
nothing. (Demo imports live inside __main__ for that reason.)

Known limitations, both measured:
- Sentence granularity is wrong for OpenAPI. Chunks are minified JSON, so
  splitting on periods cuts inside description strings and the output is not
  valid JSON. Whether the composition LLM cares is an open question.
- Fixed top-k over-selects when scores are sharply peaked. On a Spotify
  endpoint the top token scored 0.65 and the rest clustered at 0.16-0.25,
  yet k=5 pulled in four irrelevant parameters. A relative threshold would
  also be more stable: near-ties at the k boundary resolve differently on
  CPU vs MPS.

Demo: python -m attentionrag.compression   (from src/)
"""
import re


def split_into_sentences(text: str):
    """Returns (sentence, char_start, char_end) tuples."""
    sentences = []
    pos = 0
    for match in re.finditer(r"[^.!?]*[.!?]+(?:\s+|$)", text):
        start, end = match.start(), match.end()
        if text[start:end].strip():
            sentences.append((text[start:end].strip(), start, end))
        pos = end
    if pos < len(text) and text[pos:].strip():
        sentences.append((text[pos:].strip(), pos, len(text)))
    return sentences


def compress_chunk(chunk: str, token_scores: list, k: int = 3):
    """Returns (compressed_text, top_k_entries)."""
    top_k = sorted(token_scores, key=lambda t: -t["score"])[:k]
    top_k_spans = [(t["char_start"], t["char_end"]) for t in top_k]

    kept = [
        sentence
        for sentence, s_start, s_end in split_into_sentences(chunk)
        if any(s_start <= tok_start < s_end for tok_start, _ in top_k_spans)
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
