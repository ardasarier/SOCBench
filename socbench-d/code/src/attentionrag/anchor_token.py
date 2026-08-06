"""
Step 3.2a of Algorithm 1: the anchor token.

Given a chunk, the query and the hint prefix, generate exactly one token. Its
attention back to the context (Eq. 2) is what drives compression. A generated
"none" marks the chunk as irrelevant (Algorithm 1, lines 12-13).

Known limitation: the "none" gate is unreliable. It hallucinates plausible
continuations for clearly irrelevant chunks at both 0.5B and 1.5B, and flips
between CPU and MPS because it is an argmax over the vocabulary. Attention
magnitude separates relevant from irrelevant chunks far more robustly
(~11x lower peak) and is a candidate replacement.

Demo: python -m attentionrag.anchor_token   (from src/)
"""
import torch

from .models import DEVICE, compress_model as model, compress_tokenizer as tokenizer

# Verbatim from Appendix B.3. Note "prefix_hint none" near the end is literal
# text in the paper, not a format slot -- the model reads a meaningless token
# at exactly the point it is told how to refuse, which may explain the gate's
# unreliability.
ANCHOR_TOKEN_TEMPLATE = """You will be given a long context begin with 'Context:', a question begin with 'Question:', and a hint begin with 'Hint:'. Please answer the question.

Context: {chunk}

Hint: You should answer begin with {prefix_hint}, if there is no useful information in the context for the question in the context and you really don't know the answer, just answer prefix_hint none.

Question: {question}

Answer: {prefix_hint}"""


def generate_anchor_token(chunk: str, question: str, prefix_hint: str) -> str:
    prompt_text = ANCHOR_TOKEN_TEMPLATE.format(
        chunk=chunk, question=question, prefix_hint=prefix_hint
    )

    # Raw completion, not apply_chat_template: the paper's prompt ends mid-answer
    # ("Answer: {prefix_hint}") for the model to continue, not as a chat turn.
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1,  # exactly one anchor token per chunk
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    from .hint_prefix import generate_answer_hint_prefix

    question = "Where did the cat sit?"
    hint = generate_answer_hint_prefix(question)
    print(f"Question:    {question}")
    print(f"Hint prefix: {hint!r}\n")

    # Second chunk is deliberately unrelated, to exercise the "none" gate.
    chunks = [
        "The cat sat on the warm windowsill in the afternoon sun.",
        "The stock market fell sharply yesterday amid inflation fears.",
    ]
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk!r}")
        print(f"  Anchor token: {generate_anchor_token(chunk, question, hint)!r}\n")
