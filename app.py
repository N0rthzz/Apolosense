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
    /* นำ Font Google มาใช้ */
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Kanit', sans-serif;
    }

    .main {
        background-color: #f8f9fa;
    }

    /* ตกแต่ง Card ผลลัพธ์ */
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
# 3. NLP FUNCTIONS (Logic เดิมของคุณ)
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
# 4. LOAD MODEL
# ==========================================
@st.cache_resource(show_spinner=False)
def load_models():
    # หมายเหตุ: ตรวจสอบชื่อไฟล์ให้ตรงกับที่คุณมี
    model = joblib.load("apolosense_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# ==========================================
# 5. MAIN UI - ส่วนหัวหน้าเว็บ
# ==========================================
st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
st.title("Apolosense")
st.markdown("##### ระบบวิเคราะห์ระดับความจริงใจในคำขอโทษด้วย Machine Learning")
st.info("💡 **Tips:** คำขอโทษที่จริงใจมักประกอบด้วยการยอมรับผิดและการเสนอทางแก้ไข (Accountability)")

# ส่วนรับ Input
with st.container():
    user_input = st.text_area(
        "ระบุข้อความที่ต้องการวิเคราะห์:",
        height=150,
        placeholder="ตัวอย่าง: ผมขอโทษจริงๆ ครับที่ทำของพัง ผมจะรีบดำเนินการซ่อมแซมให้ทันที...",
        help="พิมพ์หรือวางข้อความคำขอโทษภาษาไทยที่นี่"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 เริ่มการวิเคราะห์", use_container_width=True, type="primary")

# ==========================================
# 6. PREDICTION & DISPLAY
# ==========================================
if analyze_button:
    if not user_input.strip():
        st.error("กรุณาป้อนข้อความก่อนกดวิเคราะห์")
    else:
        with st.spinner("🧠 AI กำลังประมวลผลโครงสร้างภาษา..."):
            try:
                # การเตรียมข้อมูล
                tokens = tokenize(user_input)
                processed_text = " ".join(tokens)
                text_vector = vectorizer.transform([processed_text])
                extra = extra_features([user_input])
                final_feature = hstack([text_vector, extra])

                # การพยากรณ์
                model, vectorizer = load_models()
                prediction_val = model.predict(final_feature)[0]
                
                # แสดงผลลัพธ์แบบสวยงาม
                st.divider()
                
                if prediction_val == 1:
                    st.markdown(f"""
                    <div class="result-card sincere-card">
                        <h1 style="font-size: 4rem;">😊</h1>
                        <h2>ผลการวิเคราะห์: มีความจริงใจสูง</h2>
                        <p>ข้อความนี้มีการแสดงความรับผิดชอบหรือความรู้สึกผิดอย่างชัดเจน</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class="result-card insincere-card">
                        <h1 style="font-size: 4rem;">🙄</h1>
                        <h2>ผลการวิเคราะห์: ฟังดูไม่จริงใจ</h2>
                        <p>ข้อความอาจมีการใช้คำเลี่ยงบาลี (Hedging) หรือขาดการแสดงความรับผิดชอบที่เพียงพอ</p>
                    </div>
                    """, unsafe_allow_html=True)

                # ส่วนสถิติ (Metrics)
                st.markdown("### 📊 รายละเอียดตัวบ่งชี้")
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("คำขอโทษที่พบ", f"{int(extra[0][3])} คำ")
                    st.markdown('</div>', unsafe_allow_html=True)
                with m_col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("การแสดงความรับผิดชอบ", f"{int(extra[0][4])} จุด")
                    st.markdown('</div>', unsafe_allow_html=True)
                with m_col3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("คำกั๊ก/เลี่ยง", f"{int(extra[0][5])} คำ")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Expansion สำหรับ Advanced Data
                with st.expander("🛠 ข้อมูลทางเทคนิคสำหรับนักวิเคราะห์"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**การตัดคำ (Tokenization):**")
                        st.caption(" / ".join(tokens))
                    with c2:
                        feat_df = pd.DataFrame({
                            'Feature': ['Sentiment', 'Length', 'Tokens', 'Apology', 'Responsible', 'Hedge'],
                            'Value': extra[0]
                        })
                        st.dataframe(feat_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ ระบบขัดข้อง: {e}")

# Footer
st.markdown("---")
st.markdown("<center style='color: gray;'>Apolosense Project © 2026 | Powered by PyThaiNLP & Scikit-learn</center>", unsafe_allow_html=True)