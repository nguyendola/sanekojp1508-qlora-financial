import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "sanekojp1508/qlora-financial"

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def preprocess_text(text: str) -> str:
    return text.strip()


def predict_sentiment(text, model, tokenizer, device, id2label):
    processed_text = preprocess_text(text)

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_id].item()

    return id2label[pred_id], confidence


def batch_predict_sentiment(texts, model, tokenizer, device, id2label):
    processed_texts = [
        preprocess_text(text)
        for text in texts
        if text.strip()
    ]

    inputs = tokenizer(
        processed_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1)

    results = []

    for i, pred_id in enumerate(pred_ids):
        pred_id = pred_id.item()
        confidence = probs[i][pred_id].item()

        results.append({
            "text": processed_texts[i],
            "label": id2label[pred_id],
            "confidence": round(confidence, 6),
        })

    return results


def render_sentiment_result(label, score):
    if label == "positive":
        st.success("😊 Positive")
    elif label == "negative":
        st.error("😠 Negative")
    else:
        st.info("😐 Neutral")

    st.write(f"**Confidence:** {score:.4f}")


st.set_page_config(
    page_title="Financial Sentiment QLoRA",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Financial Sentiment QLoRA")
st.caption("Nhập câu tài chính tiếng Anh, bấm Submit để dự đoán sentiment.")

with st.spinner("Đang tải model..."):
    model, tokenizer, device = load_model_and_tokenizer()

mode = st.radio(
    "Chế độ dự đoán",
    ["Một câu", "Nhiều câu"],
    horizontal=True,
)

if mode == "Một câu":
    text = st.text_area(
        "Nhập nội dung",
        height=180,
        placeholder="Ví dụ: The company reported strong revenue growth this quarter.",
    )

    if st.button("Submit", type="primary"):
        if not text.strip():
            st.warning("Vui lòng nhập nội dung trước khi dự đoán.")
        else:
            with st.spinner("Đang phân tích cảm xúc..."):
                label, score = predict_sentiment(
                    text,
                    model,
                    tokenizer,
                    device,
                    id2label,
                )

            render_sentiment_result(label, score)

            with st.expander("Chi tiết"):
                st.json({
                    "text": text,
                    "label": label,
                    "confidence": round(score, 6),
                    "model": MODEL_ID,
                    "device": str(device),
                })

else:
    text_batch = st.text_area(
        "Nhập nhiều câu, mỗi câu một dòng",
        height=220,
        placeholder=(
            "The company reported strong revenue growth this quarter.\n"
            "Operating profit declined compared with last year.\n"
            "The company announced a new office in Finland."
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
                    device,
                    id2label,
                )

            st.subheader("Kết quả")

            for item in results:
                st.write(f"**Sentence:** {item['text']}")
                render_sentiment_result(
                    item["label"],
                    item["confidence"],
                )
                st.divider()

            with st.expander("Chi tiết JSON"):
                st.json({
                    "results": results,
                    "model": MODEL_ID,
                    "device": str(device),
                })