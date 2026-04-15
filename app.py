import streamlit as st
import joblib 
from pythainlp.tokenize import word_tokenize

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Must be the very first st command)
# ==========================================
st.set_page_config(
    page_title="Apolosense AI | ✨ Sincerity Analyzer",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Ultra-Modern & "Over-the-Top" CSS ปรับ UI ถล่มทลาย
# ==========================================
st.markdown("""
    <style>
    /* ปรับพื้นหลังหลักของแอป */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* ซ่อน Footer และเมนู Hamburger */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ตกแต่ง Sidebar ให้ดูแพง */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        color: #ffffff;
        border-right: 1px solid #333;
    }
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stImage {
        border-radius: 50%;
        border: 3px solid #1E90FF;
        padding: 5px;
        background: #fff;
    }

    /* ตกแต่งเนื้อหาหลักให้อยู่ใน Card ลอยตัว */
    .reportview-container .main .block-container {
        background: #ffffff;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* หัวข้อหลักแบบ Gradient Text */
    .gradient-text {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
        font-size: 3.5rem !important;
        background: linear-gradient(45deg, #1E90FF, #FF1493, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        text-align: center;
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2.5rem;
    }

    /* ตกแต่งกล่องข้อความ */
    .stTextArea textarea {
        font-size: 16px !important;
        border-radius: 12px !important;
        border: 2px solid #E0E0E0;
        transition: 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #1E90FF;
        box-shadow: 0 0 10px rgba(30,144,255,0.2);
    }

    /* ตกแต่งปุ่มกดขั้นสุด (Gradient, Glow, Animation) */
    .stButton>button {
        background: linear-gradient(45deg, #1E90FF, #FF1493);
        color: white;
        border-radius: 30px;
        font-weight: 900;
        font-size: 20px;
        padding: 12px 30px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30,144,255,0.4);
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #FF1493, #1E90FF);
        box-shadow: 0 6px 20px rgba(255,20,147,0.6);
        transform: translateY(-2px) scale(1.02);
        color: white;
    }
    .stButton>button:active {
        transform: translateY(1px) scale(0.98);
    }

    /* -------------------------------------- */
    /* ลดขนาดส่วนแสดงผลลัพธ์แบบ Dynamic */
    /* -------------------------------------- */
    .result-container {
        padding: 1.2rem; /* ลดขนาดกรอบ */
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .result-icon {
        font-size: 3rem; /* ลดขนาดไอคอนจาก 5rem */
        margin-bottom: 0.2rem;
    }
    .result-label {
        font-size: 1.4rem; /* ลดขนาดตัวอักษรหลักจาก 1.8rem */
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* สีสำหรับผลลัพธ์ */
    .sincere {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    }
    .insincere {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }

    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🚨 ฟังก์ชันตัดคำ (ต้องมีเพื่อให้โหลด Vectorizer ได้)
# ==========================================
def tokenize(text):
    return word_tokenize(text, engine='newmm')

# โหลดโมเดลและ Vectorizer แบบมี Spinner สวยๆ
@st.cache_resource(show_spinner=False)
def load_models():
    model = joblib.load("apolosense_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl") 
    # สร้าง Dictionary สำหรับแปลง Output
    prediction_map = {
        0: "ฟังดูไม่จริงใจเลย",
        1: "จริงใจนะ"
    }
    return model, vectorizer, prediction_map

# ==========================================
# ส่วนออกแบบหน้าตาเว็บ (UI Layout)
# ==========================================

# --- แถบด้านข้าง (Sidebar) ---
with st.sidebar:
    # ใส่รูปภาพประกอบแบบมีกรอบขาว
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=120)
    st.markdown("<h1 style='text-align: center;'>⚙️ SYSTEM PANEL</h1>", unsafe_allow_html=True)
    
    # ย้ายเช็คสถานะโมเดลมาไว้แถบข้าง
    with st.spinner("🧠 Preparing AI Brain..."):
        try:
            model, vectorizer, prediction_map = load_models()
            st.success("✅ โมเดลพร้อมใช้งาน")
        except Exception as e:
            st.error(f"❌ โหลดโมเดลไม่สำเร็จ\n{e}")
            st.stop()
            
    st.markdown("---")
    st.markdown("💡 **เกี่ยวกับแอปพลิเคชัน**")
    st.caption("Apolosense : ระบบผู้ช่วยอัจฉริยะสำหรับวิเคราะห์ความจริงใจและน้ำเสียงในข้อความภาษาไทย ขับเคลื่อนด้วย Machine Learning รุ่นล่าสุด")
    st.caption("© 2026 Apolosense.")

# --- พื้นที่หลัก (Main Content) ---
# หัวข้อหลักแบบ Gradient Text
st.markdown('<div class="gradient-text">Apolosense </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">ผู้ช่วย AI สำหรับวิเคราะห์ความจริงใจของคำขอโทษในข้อความภาษาไทย 🇹🇭</div>', unsafe_allow_html=True)

# --------------------------------------
# ลดขนาดกล่องรับข้อมูล (ปรับ height เหลือ 80)
# --------------------------------------
user_input = st.text_area("📝 ระบุข้อความที่ต้องการวิเคราะห์:", height=80, placeholder="พิมพ์หรือวางข้อความของคุณลงในช่องนี้ เพื่อให้ AI วิเคราะห์ความจริงใจ...")

# สร้างคอลัมน์เพื่อจัดปุ่มให้อยู่ตรงกลางและเต็มหน้ากว้าง
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    analyze_button = st.button("🚀 ประมวลผลข้อความ", use_container_width=True, type="primary")

st.markdown("<br><hr>", unsafe_allow_html=True) # เว้นบรรทัดและเส้นกั้น

# ==========================================
# --- ส่วนประมวลผลและแสดงผลลัพธ์ ---
# ==========================================
if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠️ กรุณาพิมพ์ข้อความก่อนกดประมวลผลครับ")
    else:
        # แสดง Spinner แบบทันสมัย
        with st.spinner("⏳ AI กำลังทำงานขั้นสูง วิเคราะห์โครงสร้างข้อความ..."):
            try:
                # 1. ตัดคำภาษาไทย
                tokens = word_tokenize(user_input, engine='newmm')
                processed_text = " ".join(tokens) 
                
                # 2. แปลงข้อความเป็นตัวเลข (Vector)
                text_vector = vectorizer.transform([processed_text])
                
                # 3. นำเข้าโมเดลเพื่อทำนาย
                prediction_val = model.predict(text_vector)[0]
                
                # 🎯 แปลง Output (Functionality)
                # ดึงข้อความจาก Dictionary
                final_output_text = prediction_map.get(prediction_val, "未知 (ไม่รู้จัก)")
                
                # ==========================================
                # 🎯 แสดงผลลัพธ์แบบ Dynamic & Over-the-Top (Aesthetic)
                # ==========================================
                st.markdown("### 🎯 ผลการวิเคราะห์:")
                
                # ตรวจสอบค่าผลลัพธ์เพื่อเลือกสีและไอคอน
                if prediction_val == 1:
                    result_class = "sincere"
                    result_icon = "😊✨"
                else:
                    result_class = "insincere"
                    result_icon = "😒❌"
                
                # สร้างแผงแสดงผลลัพธ์แบบ HTML
                result_panel_html = f"""
                    <div class="result-container {result_class}">
                        <div class="result-icon">{result_icon}</div>
                        <div class="result-label">{final_output_text}</div>
                        <p style="font-size: 0.8rem; opacity: 0.8; margin-top: 5px;">(ผลลัพธ์อิงตามข้อมูลที่ใช้เทรนโมเดล)</p>
                    </div>
                """
                st.markdown(result_panel_html, unsafe_allow_html=True)
                
                # ส่วนขยายดูเบื้องหลัง (ซ่อนไว้เริ่มต้น)
                with st.expander("🔍 ดูข้อมูลดิบเบื้องหลังการทำงาน (หลังบ้าน)"):
                    st.write("ระบบได้ทำการตัดคำข้อความของคุณดังนี้:")
                    st.code(tokens, language='json')
                    st.write(f"ค่าผลลัพธ์ดิบจากโมเดล: `{prediction_val}`")
                    
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดร้ายแรงระหว่างรันโมเดล: {e}")