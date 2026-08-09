import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VisionAI | Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

/* Header */
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid #334155;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    color: #94a3b8;
    font-size: 17px;
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 16px;
    background: #111827;
    border: 1px solid #334155;
    margin-bottom: 15px;
}

.card-title {
    font-size: 18px;
    font-weight: 700;
}

.stat {
    padding: 18px;
    border-radius: 14px;
    background: #111827;
    border: 1px solid #334155;
    text-align: center;
}

.stat-icon {
    font-size: 28px;
}

.stat-value {
    font-size: 25px;
    font-weight: bold;
}

.stat-label {
    color: #94a3b8;
    font-size: 14px;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    font-weight: 700;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    st.markdown("---")

    confidence = st.slider(
        "🎯 Confidence",
        0.1,
        1.0,
        0.45,
        0.05
    )

    st.markdown("---")

    st.markdown("### 🧠 Model")

    st.info(
        "YOLOv8 Nano\n\n"
        "⚡ Fast inference\n\n"
        "💻 CPU friendly"
    )

    st.markdown("---")

    st.markdown(
        """
        ### 🛠️ Technology

        🧠 YOLOv8  
        👁️ Computer Vision  
        🐍 Python  
        🎨 Streamlit  
        📷 OpenCV
        """
    )


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🎯 VisionAI</h1>

<p>
Real-Time Object Detection powered by
<b>YOLOv8 + OpenCV + Streamlit</b>
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODE SELECTION
# =========================================================

st.markdown("### 🚀 Detection Mode")

mode = st.radio(
    "",
    [
        "📤 Upload Image",
        "📷 Camera Capture"
    ],
    horizontal=True
)


# =========================================================
# DETECTION FUNCTION
# =========================================================

def detect_image(image):

    image_np = np.array(image)

    results = model.predict(
        image_np,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    # Count objects
    counts = {}

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            counts[class_name] = (
                counts.get(class_name, 0) + 1
            )

    total_objects = sum(counts.values())

    return annotated, counts, total_objects


# =========================================================
# UPLOAD IMAGE
# =========================================================

if mode == "📤 Upload Image":

    st.markdown(
        """
        <div class="card">
        <div class="card-title">📤 Upload your image</div>
        <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        if st.button("🔍 Detect Objects"):

            with st.spinner("🧠 AI is analyzing the image..."):

                annotated, counts, total = detect_image(image)

            st.success("✅ Detection completed successfully!")

            # ---------------------------
            # Images
            # ---------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 🖼️ Original")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.markdown("### 🎯 Detection Result")

                st.image(
                    annotated,
                    channels="BGR",
                    use_container_width=True
                )

            # ---------------------------
            # Stats
            # ---------------------------

            st.markdown("### 📊 Detection Statistics")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="stat">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Objects Detected</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="stat">
                    <div class="stat-icon">🏷️</div>
                    <div class="stat-value">{len(counts)}</div>
                    <div class="stat-label">Unique Classes</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:

                st.markdown(
                    f"""
                    <div class="stat">
                    <div class="stat-icon">🧠</div>
                    <div class="stat-value">YOLOv8</div>
                    <div class="stat-label">AI Model</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ---------------------------
            # Classes
            # ---------------------------

            if counts:

                st.markdown("### 🔎 Detected Objects")

                for name, count in counts.items():

                    st.write(
                        f"🔹 **{name}** — `{count}` detected"
                    )

        else:

            st.info(
                "👆 Upload an image and click **Detect Objects**."
            )


# =========================================================
# CAMERA
# =========================================================

else:

    st.markdown(
        """
        <div class="card">
        <div class="card-title">📷 Camera Detection</div>
        <p>Take a picture using your device camera.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    camera_image = st.camera_input(
        "Take a picture",
        label_visibility="collapsed"
    )

    if camera_image:

        image = Image.open(camera_image).convert("RGB")

        with st.spinner("🧠 Detecting objects..."):

            annotated, counts, total = detect_image(image)

        st.success("✅ Detection completed!")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 📷 Captured")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### 🎯 AI Detection")

            st.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        st.markdown("### 📊 Results")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "🎯 Objects",
                total
            )

        with c2:
            st.metric(
                "🏷️ Classes",
                len(counts)
            )

        if counts:

            for name, count in counts.items():

                st.write(
                    f"🔹 **{name}** — {count}"
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <b>VisionAI</b>  
Powered by YOLOv8 • OpenCV • Streamlit

</div>
""", unsafe_allow_html=True)
