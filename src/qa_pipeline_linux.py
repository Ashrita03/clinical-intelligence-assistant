import re
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=1)
def load_qa_model():
    """Load a smaller Qwen model for CPU inference."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def generate_grounded_answer(
    question,
    context,
    tokenizer,
    model,
    conversation_history=None,
):
    if not question or not question.strip():
        return "Please enter a question."

    if not context or not context.strip():
        return "No usable report context was found."

    system_prompt = (
        "You are an educational medical report assistant. "
        "Use the supplied report as the only source of patient-specific facts. "
        "Do not invent findings, diagnoses, or treatments. "
        "Answer the question concisely. "
        "Distinguish general education from report findings."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        for item in conversation_history[-6:]:
            if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
                messages.append({
                    "role": item["role"],
                    "content": str(item.get("content", "")),
                })

    messages.append({
        "role": "user",
        "content": (
            f"REPORT CONTEXT:\n{context}\n\n"
            f"CURRENT QUESTION:\n{question}\n\n"
            "Answer using the report. Acknowledge missing information."
        ),
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer or "I could not generate a reliable answer."


def build_context_from_results(retrieval_results):
    if not retrieval_results:
        return ""

    chunks = []
    for result in retrieval_results:
        chunk = result.get("chunk", "").strip()
        if chunk and chunk not in chunks:
            chunks.append(chunk)

    return "\n\n".join(chunks)
