import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "sanekojp1508/cyber_abte"

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


def render_sentiment_result(label, score):
    if label == "positive":
        st.success(f"😊 Positive")
    elif label == "negative":
        st.error(f"😠 Negative")
    else:
        st.info(f"😐 Neutral")

    st.write(f"**Confidence:** {score:.4f}")


st.set_page_config(
    page_title="Vietnamese Sentiment Demo",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Vietnamese Sentiment Analysis")
st.caption("Nhập nội dung tiếng Việt, bấm Submit để dự đoán cảm xúc.")

with st.spinner("Đang tải model..."):
    model, tokenizer, device = load_model_and_tokenizer()

text = st.text_area(
    "Nhập nội dung",
    height=180,
    placeholder="Ví dụ: Bài viết này rất hữu ích và tích cực",
)

if st.button("Submit", type="primary"):
    if not text.strip():
        st.warning("Vui lòng nhập nội dung trước khi dự đoán.")
    else:
        with st.spinner("Đang phân tích cảm xúc..."):
            label, score = predict_sentiment(text, model, tokenizer, device, id2label)

        render_sentiment_result(label, score)

        with st.expander("Chi tiết"):
            st.json(
                {
                    "text": text,
                    "label": label,
                    "confidence": round(score, 6),
                    "model": MODEL_ID,
                    "device": str(device),
                }
            )