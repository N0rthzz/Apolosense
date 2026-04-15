import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from pythainlp.tokenize import word_tokenize

# ==========================================
# 1. Page Config
# ==========================================
st.set_page_config(
    page_title="Apolosense AI",
    page_icon="✨",
    layout="wide", # ใช้ Wide mode เพื่อให้จัดวาง Column ได้สวยขึ้น
)

# ==========================================
# 2. Optimized CSS (Minimalist Style)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
        font-size: 15px;
    }

    /* กระชับพื้นที่ Header */
    .block-container {
        padding-top: 2rem !important;
        max-width: 900px;
    }

    /* ผลลัพธ์ขนาดพอดี (Compact Result) */
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

    /* ปรับแต่ง Metric ให้ดูทันสมัย */
    .stMetric {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. NLP FUNCTIONS (คงเดิม)
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
            pos - neg,
            len(t),
            len(tokenize(t)),
            sum(t.count(w) for w in apology_words),
            sum(t.count(w) for w in responsible_words),
            sum(t.count(w) for w in hedging_words)
        ])
    return np.array(features)

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        return joblib.load("apolosense_model.pkl"), joblib.load("vectorizer.pkl")
    except:
        return None, None

model, vectorizer = load_models()

# ==========================================
# 4. MAIN UI (Compact Layout)
# ==========================================
# ส่วนหัวแบบบรรทัดเดียว
head_col1, head_col2 = st.columns([1, 6])
with head_col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=60)
with head_col2:
    st.title("Apolosense")
    st.caption("AI วิเคราะห์ระดับความจริงใจในคำขอโทษ 🇹🇭")

st.write("") # Spacer

# แบ่งหน้าจอเป็น 2 ฝั่ง (Input | Result)
main_col1, main_col2 = st.columns([1, 1], gap="large")

with main_col1:
    st.markdown("**📝 ข้อความที่ต้องการวิเคราะห์**")
    user_input = st.text_area(
        label="input_area",
        label_visibility="collapsed",
        height=180,
        placeholder="พิมพ์หรือวางคำขอโทษที่นี่...",
    )
    
    if st.button("🚀 วิเคราะห์ทันที", use_container_width=True, type="primary"):
        do_analyze = True
    else:
        do_analyze = False

with main_col2:
    if do_analyze:
        if not user_input.strip():
            st.warning("⚠️ กรุณากรอกข้อความ")
        elif model is None:
            st.error("❌ ไม่พบโมเดล")
        else:
            with st.spinner("กำลังประมวลผล..."):
                # Processing
                tokens = tokenize(user_input)
                processed_text = " ".join(tokens)
                text_vector = vectorizer.transform([processed_text])
                extra = extra_features([user_input])
                final_feature = hstack([text_vector, extra])
                prediction_val = model.predict(final_feature)[0]

                # Result Box
                if prediction_val == 1:
                    st.markdown(f"""
                    <div class="result-box sincere-res">
                        <span style="font-size: 30px;">😊</span>
                        <div>
                            <b style="font-size: 18px;">ผลลัพธ์: มีความจริงใจ</b><br>
                            <small>โครงสร้างภาษาดูน่าเชื่อถือและรับผิดชอบ</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box insincere-res">
                        <span style="font-size: 30px;">😒</span>
                        <div>
                            <b style="font-size: 18px;">ผลลัพธ์: ฟังดูไม่จริงใจ</b><br>
                            <small>มีการใช้คำเลี่ยงบาลีหรือขาดน้ำหนักของการกระทำ</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Metrics (Compact Grid)
                st.write("**📊 สรุปข้อมูลที่พบ**")
                m_c1, m_c2 = st.columns(2)
                m_c1.metric("คำขอโทษ", f"{int(extra[0][3])}")
                m_c1.metric("คำรับผิดชอบ", f"{int(extra[0][4])}")
                m_c2.metric("คำกั๊ก/เลี่ยง", f"{int(extra[0][5])}")
                m_c2.metric("คะแนนอารมณ์", f"{int(extra[0][0])}")

                with st.expander("🛠 ข้อมูลเทคนิค"):
                    st.json({"tokens": tokens[:10], "features_count": int(final_feature.shape[1])})
    else:
        # แสดง Placeholder เมื่อยังไม่มีการกดปุ่ม
        st.info("💡 **คำแนะนำ**\n\nคำขอโทษที่ดีควรบอกสิ่งที่ทำผิดชัดเจน และมีแนวทางการแก้ไข (Action Plan) AI จะจับจุดเหล่านี้จากฐานข้อมูลภาษาไทย")

# Footer ชิดขอบล่าง
st.markdown("---")
st.markdown("<div style='text-align: right; color: #bdc3c7; font-size: 0.8rem;'>Apolosense Project v2.0 | 2026</div>", unsafe_allow_html=True)