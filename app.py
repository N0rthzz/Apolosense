import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from pythainlp.tokenize import word_tokenize

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Apolosense AI",
    page_icon="✨",
    layout="wide",
)

# ==========================================
# 2. CSS
# ==========================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem !important;
        max-width: 900px;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 20px;
    }

    .sincere-res {
        background-color: #e6f7ed;
        border: 1px solid #2ecc71;
        color: #1e8449;
    }

    .insincere-res {
        background-color: #fdf2f2;
        border: 1px solid #e74c3c;
        color: #922b21;
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


def extra_features(texts):
    features = []

    for t in texts:
        pos = sum(t.count(w) for w in positive_words)
        neg = sum(t.count(w) for w in negative_words)

        features.append([
            pos - neg,                              # 1 sentiment
            len(t),                                 # 2 char length
            len(tokenize(t)),                       # 3 token length
            sum(t.count(w) for w in apology_words), # 4 apology
            sum(t.count(w) for w in responsible_words), # 5 responsibility
            sum(t.count(w) for w in hedging_words)  # 6 hedge
        ])

    return np.array(features)


# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource(show_spinner=False)
def load_models():
    model = joblib.load("apolosense_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_models()
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ==========================================
# 5. SESSION STATE
# ==========================================
if "do_analyze" not in st.session_state:
    st.session_state.do_analyze = False

# ==========================================
# 6. HEADER
# ==========================================
head_col1, head_col2 = st.columns([1, 6])
with head_col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=60)
with head_col2:
    st.title("Apolosense")
    st.caption("AI วิเคราะห์ระดับความจริงใจในคำขอโทษ 🇹🇭")

# ==========================================
# 7. MAIN LAYOUT
# ==========================================
main_col1, main_col2 = st.columns([1, 1], gap="large")

with main_col1:
    st.markdown("**📝 ข้อความที่ต้องการวิเคราะห์**")

    user_input = st.text_area(
        label="input_area",
        label_visibility="collapsed",
        height=180,
        placeholder="พิมพ์หรือวางคำขอโทษที่นี่..."
    )

    if st.button("🚀 วิเคราะห์ทันที", use_container_width=True, type="primary"):
        st.session_state.do_analyze = True

with main_col2:
    if st.session_state.do_analyze:
        if not user_input.strip():
            st.warning("⚠️ กรุณากรอกข้อความ")
        else:
            with st.spinner("กำลังประมวลผล..."):
                try:
                    tokens = tokenize(user_input)
                    processed_text = " ".join(tokens)

                    # TF-IDF
                    text_vector = vectorizer.transform([processed_text])

                    # Extra features = 6
                    extra = extra_features([user_input])

                    # Final shape should be 5006
                    final_feature = hstack([text_vector, extra])

                    prediction_val = model.predict(final_feature)[0]

                    if prediction_val == 1:
                        st.markdown("""
                        <div class="result-box sincere-res">
                            <span style="font-size: 30px;">😊</span>
                            <div>
                                <b style="font-size: 18px;">ผลลัพธ์: มีความจริงใจ</b><br>
                                <small>โครงสร้างภาษาดูน่าเชื่อถือและรับผิดชอบ</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-box insincere-res">
                            <span style="font-size: 30px;">😒</span>
                            <div>
                                <b style="font-size: 18px;">ผลลัพธ์: ฟังดูไม่จริงใจ</b><br>
                                <small>มีการใช้คำเลี่ยงหรือขาดน้ำหนักของการกระทำ</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("**📊 สรุปข้อมูลที่พบ**")
                    m_c1, m_c2 = st.columns(2)
                    m_c1.metric("คำขอโทษ", f"{int(extra[0][3])}")
                    m_c1.metric("คำรับผิดชอบ", f"{int(extra[0][4])}")
                    m_c2.metric("คำกั๊ก/เลี่ยง", f"{int(extra[0][5])}")
                    m_c2.metric("คะแนนอารมณ์", f"{int(extra[0][0])}")

                    with st.expander("🛠 ข้อมูลเทคนิค"):
                        st.json({
                            "tokens_preview": tokens[:10],
                            "feature_count": int(final_feature.shape[1]),
                            "expected": 5006
                        })

                except Exception as e:
                    st.error(f"❌ Prediction Error: {e}")
    else:
        st.info("💡 คำขอโทษที่ดีควรระบุสิ่งที่ผิด + การแก้ไขอย่างชัดเจน")

# ==========================================
# 8. FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: right; color: gray; font-size: 0.8rem;'>Apolosense Project v2.1 | 2026</div>",
    unsafe_allow_html=True
)