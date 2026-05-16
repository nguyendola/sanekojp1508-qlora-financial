import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sanekojp1508/qlora-financial-merged"

USER_PROMPT_TEMPLATE = """Predict the sentiment of the following input sentence.
The response must begin with "Sentiment: ", followed by one of these keywords: "positive", "negative", or "neutral", to reflect the sentiment of the input sentence.

Sentence: {input}"""


@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()

    return model, tokenizer


def preprocess_text(text: str) -> str:
    return text.strip()


def extract_sentiment(text: str) -> str:
    text = text.strip().lower()

    # Ưu tiên bắt đúng format: Sentiment: neutral
    match = re.search(r"sentiment\s*:\s*(positive|negative|neutral)", text)

    if match:
        return match.group(1)

    # Nếu model chỉ trả về 1 từ
    words = re.findall(r"\b(positive|negative|neutral)\b", text)

    if words:
        return words[-1]  # lấy nhãn cuối cùng thay vì positive đầu tiên

    return "unknown"


def predict_sentiment(text, model, tokenizer):
    processed_text = preprocess_text(text)

    user_prompt = USER_PROMPT_TEMPLATE.format(input=processed_text)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You must fulfill the user request.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    input_prompt = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = output_ids[:, inputs["input_ids"].shape[-1]:]

    output_text = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()

    label = extract_sentiment(output_text)

    return label, output_text


def predict_sentiment(text, model, tokenizer):
    tokenizer.pad_token_id = tokenizer.eos_token_id

    processed_text = preprocess_text(text)

    user_prompt = USER_PROMPT_TEMPLATE.format(input=processed_text)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You must fulfill the user request."
        },
        {
            "role": "user",
            "content": user_prompt
        },
    ]

    input_prompt = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        add_special_tokens=False
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = output_ids[
        :,
        inputs["input_ids"][0].shape[-1]:output_ids.shape[-1]
    ]

    output_text = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()

    label = extract_sentiment(output_text)

    return label, output_text


def render_sentiment_result(label, output_text=None):
    if label == "positive":
        st.success("😊 Positive")
    elif label == "negative":
        st.error("😠 Negative")
    elif label == "neutral":
        st.info("😐 Neutral")
    else:
        st.warning("⚠️ Unknown")

    if output_text:
        st.write(f"**Model output:** `{output_text}`")


st.set_page_config(
    page_title="Financial Sentiment QLoRA",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Financial Sentiment QLoRA")
st.caption("Nhập câu tài chính tiếng Anh để dự đoán sentiment.")

example_texts = [
    "Operating profit increased by 25 percent.",
    "The company reported significant losses.",
    "I have no idea.",
]

with st.spinner("Đang tải model..."):
    model, tokenizer = load_model_and_tokenizer()

mode = st.radio(
    "Chế độ dự đoán",
    ["Một câu", "Nhiều câu"],
    horizontal=True,
)

if mode == "Một câu":
    st.markdown("### Ví dụ mẫu")

    cols = st.columns(3)

    for idx, example in enumerate(example_texts):
        with cols[idx]:
            if st.button(
                f"Ví dụ {idx + 1}",
                use_container_width=True,
                key=f"single_example_{idx}",
            ):
                st.session_state["example_text"] = example

    text = st.text_area(
        "Nhập nội dung",
        value=st.session_state.get("example_text", ""),
        height=180,
        placeholder="Ví dụ: The company reported strong revenue growth this quarter.",
    )

    if st.button("Submit", type="primary"):
        if not text.strip():
            st.warning("Vui lòng nhập nội dung trước khi dự đoán.")
        else:
            with st.spinner("Đang phân tích cảm xúc..."):
                label, output_text = predict_sentiment(
                    text,
                    model,
                    tokenizer,
                )

            render_sentiment_result(label, output_text)

            with st.expander("Chi tiết"):
                st.json({
                    "text": text,
                    "label": label,
                    "model_output": output_text,
                    "model": MODEL_ID,
                    "device": str(model.device),
                })

else:
    st.markdown("### Ví dụ batch")

    if st.button(
        "Load ví dụ batch",
        use_container_width=True,
        key="load_batch_examples",
    ):
        st.session_state["example_batch_text"] = "\n".join(example_texts)

    text_batch = st.text_area(
        "Nhập nhiều câu, mỗi câu một dòng",
        value=st.session_state.get("example_batch_text", ""),
        height=220,
        placeholder=(
            "Operating profit increased by 25 percent.\n"
            "The company reported significant losses.\n"
            "The company announced a new board meeting."
        ),
    )

    if st.button("Submit batch", type="primary"):
        texts = [
            line.strip()
            for line in text_batch.splitlines()
            if line.strip()
        ]

        if not texts:
            st.warning("Vui lòng nhập ít nhất một câu.")
        else:
            with st.spinner("Đang phân tích batch..."):
                results = batch_predict_sentiment(
                    texts,
                    model,
                    tokenizer,
                )

            st.subheader("Kết quả")

            for item in results:
                st.write(f"**Sentence:** {item['text']}")
                render_sentiment_result(
                    item["label"],
                    item["output"],
                )
                st.divider()

            with st.expander("Chi tiết JSON"):
                st.json({
                    "results": results,
                    "model": MODEL_ID,
                    "device": str(model.device),
                })