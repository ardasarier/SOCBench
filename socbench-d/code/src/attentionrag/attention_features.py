"""
Step 3.2b of Algorithm 1: attention features (Eq. 2).

Scores every context token by how much the anchor token attends back to it.

DESIGN DECISION -- fills a gap in the paper. Eq. 2 sums attention across all
layers but never states how the heads within a layer are combined. We average
heads first, then sum across layers. Summing across layers is the paper's own
choice (their Table 9 ablation: all layers 0.42 EM vs 0.35-0.40 for any
subset); the head averaging is ours and must be reported as such.

Demo: python -m attentionrag.attention_features   (from src/)
"""
import torch

from .anchor_token import ANCHOR_TOKEN_TEMPLATE
from .models import DEVICE, compress_model as model, compress_tokenizer as tokenizer


def compute_attention_feature(chunk: str, question: str, prefix_hint: str, anchor_token: str, layer_range=None):
    """
    Returns (token_scores, num_layers) where each score is a dict of
    token / score / char_start / char_end. Character offsets are relative to
    the chunk and are what Eq. 3 uses to map tokens back to segments.
    """
    prompt_text = ANCHOR_TOKEN_TEMPLATE.format(
        chunk=chunk, question=question, prefix_hint=prefix_hint
    )
    # The anchor token must be IN the input to have an attention row at all;
    # during generation it is only being predicted. Hence a second forward pass.
    full_text = prompt_text + anchor_token

    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoding.pop("offset_mapping")[0]  # CPU only; used for Python indexing
    encoding = encoding.to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)

    chunk_start = prompt_text.find(chunk)
    chunk_end = chunk_start + len(chunk)
    context_token_indices = [
        i for i, (start, end) in enumerate(offsets.tolist())
        if start >= chunk_start and end <= chunk_end and end > start
    ]

    anchor_position = encoding["input_ids"].shape[1] - 1

    # Eq. 2 sums over all layers. layer_range restricts it to a band, so the
    # paper's Table 9 ablation (shallow/middle/deep vs all) can be replicated
    # on structured API text rather than prose QA.
    layers = outputs.attentions
    if layer_range is not None:
        layers = layers[layer_range[0]:layer_range[1]]

    num_layers = len(layers)
    attention_feature = torch.zeros(len(context_token_indices), device=DEVICE)

    for layer_attn in layers:                       # (1, heads, seq, seq) per layer
        avg_over_heads = layer_attn[0].mean(dim=0)  # heads averaged (our choice)
        anchor_row = avg_over_heads[anchor_position]
        attention_feature += anchor_row[context_token_indices]  # summed over layers

    token_scores = []
    for local_i, token_idx in enumerate(context_token_indices):
        token_str = tokenizer.decode([encoding["input_ids"][0][token_idx]])
        char_start, char_end = offsets[token_idx].tolist()

        # BPE bundles a leading space into the following word's token, while the
        # sentence regex consumes trailing whitespace into the preceding
        # sentence. Without this shift, the first word of every sentence is
        # credited to the previous one.
        if token_str.startswith(" "):
            char_start += 1

        token_scores.append({
            "token": token_str,
            "score": attention_feature[local_i].item(),
            "char_start": char_start - chunk_start,
            "char_end": char_end - chunk_start,
        })

    return token_scores, num_layers


if __name__ == "__main__":
    from .anchor_token import generate_anchor_token
    from .hint_prefix import generate_answer_hint_prefix

    question = "Where did the cat sit?"
    hint = generate_answer_hint_prefix(question)
    print(f"Question:    {question}")
    print(f"Hint prefix: {hint!r}\n")

    chunks = [
        "The cat sat on the warm windowsill in the afternoon sun.",
        "The stock market fell sharply yesterday amid inflation fears.",
    ]
    for i, chunk in enumerate(chunks):
        anchor = generate_anchor_token(chunk, question, hint)
        print(f"Chunk {i}: {chunk!r}")
        print(f"  Anchor token: {anchor!r}")

        if anchor.strip().lower() == "none":
            print("  -> skipped (Algorithm 1, lines 12-13)\n")
            continue

        scores, num_layers = compute_attention_feature(chunk, question, hint, anchor)
        print(f"  Attention per context token (summed over {num_layers} layers):")
        for entry in sorted(scores, key=lambda x: -x["score"]):
            print(f"    {entry['token']!r:<15} {entry['score']:.4f}")
        print()
