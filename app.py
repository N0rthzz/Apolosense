import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from pythainlp.tokenize import word_tokenize

# ==========================================
# 1. หน้าเว็บ & ธีม (Modern Theme)
# ==========================================
st.set_page_config(
    page_title="Apolosense | Sincerity Analyzer",
    page_icon="✨",
    layout="centered",
)

# ==========================================
# 2. CSS - ปรับแต่ง UI ให้ดูหรูหราขึ้น
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Kanit', sans-serif;
    }
    .main {
        background-color: #f8f9fa;
    }
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        transition: transform 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .sincere-card {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        box-shadow: 0 10px 20px rgba(0,176,155,0.2);
    }
    .insincere-card {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        box-shadow: 0 10px 20px rgba(255,75,43,0.2);
    }
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
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
        features.append([
            sentiment_score(t),
            len(t),
            len(tokenize(t)),
            keyword_count(t, apology_words),
            keyword_count(t, responsible_words),
            keyword_count(t, hedging_words)
        ])
    return np.array(features)

# ==========================================
# 4. LOAD MODEL (ย้ายมาไว้ข้างบนเพื่อให้เรียกใช้ได้ทั่วถึง)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        model = joblib.load("apolosense_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"ไม่สามารถโหลดไฟล์โมเดลได้: {e}")
        return None, None

# โหลดเตรียมไว้ก่อนเลย
model, vectorizer = load_models()

# ==========================================
# 5. MAIN UI
# ==========================================
st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
st.title("Apolosense")
st.markdown("##### ระบบวิเคราะห์ระดับความจริงใจในคำขอโทษด้วย Machine Learning")
st.info("💡 **Tips:** คำขอโทษที่จริงใจมักประกอบด้วยการยอมรับผิดและการเสนอทางแก้ไข")

user_input = st.text_area(
    "ระบุข้อความที่ต้องการวิเคราะห์:",
    height=150,
    placeholder="พิมพ์คำขอโทษที่นี่...",
    help="ระบบจะวิเคราะห์จากโครงสร้างประโยคและคำสำคัญ"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🚀 เริ่มการวิเคราะห์", use_container_width=True, type="primary")

# ==========================================
# 6. PREDICTION & DISPLAY
# ==========================================
if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ กรุณาป้อนข้อความก่อน")
    elif model is None or vectorizer is None:
        st.error("❌ ไม่พบโมเดลในระบบ กรุณาตรวจสอบไฟล์ .pkl")
    else:
        with st.spinner("🧠 AI กำลังวิเคราะห์..."):
            try:
                # --- ส่วนสำคัญ: ดึงคุณลักษณะ (Features) ---
                tokens = tokenize(user_input)
                processed_text = " ".join(tokens)
                
                # ใช้ vectorizer ที่โหลดมาจากด้านบน
                text_vector = vectorizer.transform([processed_text])
                extra = extra_features([user_input])
                
                # รวม Features (TF-IDF + Extra Features)
                final_feature = hstack([text_vector, extra])

                # ทำนายผล
                prediction_val = model.predict(final_feature)[0]
                
                # --- การแสดงผล ---
                st.divider()
                if prediction_val == 1:
                    st.markdown("""
                    <div class="result-card sincere-card">
                        <h1 style="font-size: 4rem;">😊</h1>
                        <h2>ผลการวิเคราะห์: มีความจริงใจ</h2>
                        <p>ข้อความแสดงถึงความรับผิดชอบและเจตนาที่ดี</p>
                    </div>""", unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown("""
                    <div class="result-card insincere-card">
                        <h1 style="font-size: 4rem;">😒</h1>
                        <h2>ผลการวิเคราะห์: ฟังดูไม่จริงใจ</h2>
                        <p>ข้อความอาจขาดการแสดงความรับผิดชอบที่ชัดเจน</p>
                    </div>""", unsafe_allow_html=True)

                # แสดง Metrics
                st.markdown("### 📊 รายละเอียดตัวบ่งชี้")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("คำขอโทษ", f"{int(extra[0][3])} คำ")
                    st.markdown('</div>', unsafe_allow_html=True)
                with m_col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("ความรับผิดชอบ", f"{int(extra[0][4])} จุด")
                    st.markdown('</div>', unsafe_allow_html=True)
                with m_col3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("คำเลี่ยงบาลี", f"{int(extra[0][5])} คำ")
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("🔍 ดูข้อมูลทางเทคนิค"):
                    st.write("**Tokens:**", tokens)
                    st.write("**Feature shape:**", final_feature.shape)

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดระหว่างวิเคราะห์: {e}")

# Footer
st.markdown("---")
st.markdown("<center style='color: gray;'>Apolosense Project © 2026 | Powered by PyThaiNLP</center>", unsafe_allow_html=True)