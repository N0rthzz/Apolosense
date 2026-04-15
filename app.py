import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from pythainlp.tokenize import word_tokenize

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(
    page_title="Apolosense AI | ✨ Sincerity Analyzer",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS
# ==========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.result-container {
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    margin-top: 1rem;
    color: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.result-icon {
    font-size: 3rem;
}
.result-label {
    font-size: 1.4rem;
    font-weight: 800;
}
.sincere {
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
}
.insincere {
    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. NLP FUNCTIONS
# ==========================================
def tokenize(text):
    return word_tokenize(text, engine="newmm")


apology_words = ["ขอโทษ", "ขออภัย", "ต้องขออภัย", "เสียใจ"]
responsible_words = ["รับผิดชอบ", "ดำเนินการ", "แก้ไข", "ปรับปรุง", "ดูแลให้"]
hedging_words = ["อาจ", "อาจจะ", "น่าจะ", "คง", "เหมือนจะ"]
positive_words = ["ดี", "ดีมาก", "ยอดเยี่ยม", "ชอบ"]
negative_words = ["แย่", "แย่มาก", "ห่วย", "ผิดหวัง"]


def sentiment_score(text):
    pos = sum(text.count(w) for w in positive_words)
    neg = sum(text.count(w) for w in negative_words)
    return pos - neg


def keyword_count(text, keywords):
    return sum(text.count(w) for w in keywords)


def extra_features(texts):
    features = []

    for t in texts:
        senti = sentiment_score(t)
        length = len(t)
        apology = keyword_count(t, apology_words)
        responsible = keyword_count(t, responsible_words)
        hedge = keyword_count(t, hedging_words)

        features.append([
            senti,
            length,
            apology,
            responsible,
            hedge
        ])

    return np.array(features)


# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource(show_spinner=False)
def load_models():
    model = joblib.load("apolosense_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    prediction_map = {
        0: "ฟังดูไม่จริงใจเลย",
        1: "จริงใจนะ"
    }

    return model, vectorizer, prediction_map


# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚙️ SYSTEM PANEL")

    with st.spinner("🧠 Preparing AI Brain..."):
        try:
            model, vectorizer, prediction_map = load_models()
            st.success("✅ โมเดลพร้อมใช้งาน")
        except Exception as e:
            st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
            st.stop()

    st.markdown("---")
    st.caption("Apolosense : ระบบวิเคราะห์ความจริงใจของคำขอโทษภาษาไทย")


# ==========================================
# 6. MAIN UI
# ==========================================
st.title("🤖 Apolosense")
st.subheader("AI วิเคราะห์ความจริงใจของคำขอโทษภาษาไทย 🇹🇭")

user_input = st.text_area(
    "📝 ระบุข้อความที่ต้องการวิเคราะห์:",
    height=120,
    placeholder="พิมพ์ข้อความคำขอโทษ..."
)

analyze_button = st.button("🚀 วิเคราะห์ข้อความ", use_container_width=True)

# ==========================================
# 7. PREDICTION
# ==========================================
if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠️ กรุณาพิมพ์ข้อความก่อน")
    else:
        with st.spinner("⏳ AI กำลังวิเคราะห์..."):
            try:
                # tokenize
                tokens = tokenize(user_input)
                processed_text = " ".join(tokens)

                # TF-IDF
                text_vector = vectorizer.transform([processed_text])

                # Extra Features
                extra = extra_features([user_input])

                # Final Features
                final_feature = hstack([text_vector, extra])

                # Predict
                prediction_val = model.predict(final_feature)[0]
                final_output_text = prediction_map[prediction_val]

                st.markdown("### 🎯 ผลการวิเคราะห์")

                if prediction_val == 1:
                    result_class = "sincere"
                    result_icon = "😊✨"
                else:
                    result_class = "insincere"
                    result_icon = "😒❌"

                result_html = f"""
                <div class="result-container {result_class}">
                    <div class="result-icon">{result_icon}</div>
                    <div class="result-label">{final_output_text}</div>
                </div>
                """
                st.markdown(result_html, unsafe_allow_html=True)

                with st.expander("🔍 ดูข้อมูลเบื้องหลัง"):
                    st.write("Tokens:", tokens)
                    st.write("Feature shape:", final_feature.shape)
                    st.write("Prediction value:", prediction_val)

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {e}")