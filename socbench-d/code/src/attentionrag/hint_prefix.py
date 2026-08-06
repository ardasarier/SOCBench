"""
Step 3.1 of Algorithm 1: the answer hint prefix.

Reformulates a query into next-token-prediction form ("Where is Daniel?" ->
"Daniel is in the ___"), collapsing its semantic focus onto a single missing
token that attention can then be measured against.

Demo: python -m attentionrag.hint_prefix   (from src/)
"""
import torch

from .models import DEVICE, hint_model as model, hint_tokenizer as tokenizer

# Verbatim from Appendix B.1.
HINT_PREFIX_TEMPLATE = """You are a formatting assistant. Given a question, your task is to generate a corresponding answering format. The format should maintain the same structure as the question but transform it into an incomplete answer template. If it is impossible to generate a format, return "None".

The format is like an complete answer, but truncated before the key word, and the key word is not included in the format.

For instance, if the question is "Where is Daniel?", the format should be "Daniel is in the", as the next word is the key word.

Note: For yes/no questions, such as "Is Tom here?", return "None" because these questions are typically answered with "yes" or "no" and do not have a natural continuation that leads to a single keyword.

Examples:
1. Question: Where is Daniel?
Format: Daniel is in the

2. Question: What time is it?
Format: It is

3. Question: Who is responsible for this?
Format: The person responsible for this is

4. Question: Which film was released more recently, Dance With A Stranger or Miley Naa Miley Hum?
Format: The film released more recently

5. Question: Is Tom here?
Format: None

In generation, you should only return the format, not any other text.

Now, here's a new question:

Question: {question}
Format:"""


def generate_answer_hint_prefix(question: str) -> str:
    """Returns the hint prefix, or "None" when the question has no single
    focal continuation (yes/no questions, per Appendix B.1)."""
    messages = [{"role": "user", "content": HINT_PREFIX_TEMPLATE.format(question=question)}]

    # return_dict=True keeps input_ids and attention_mask together so **inputs
    # passes both to generate(); without it some transformers versions return a
    # bare tensor and generate() fails looking for .shape on a BatchEncoding.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    for q in ["Where did the cat sit?", "Is Tom here?", "What time is it?"]:
        print(f"Question:    {q}")
        print(f"Hint prefix: {generate_answer_hint_prefix(q)!r}\n")
