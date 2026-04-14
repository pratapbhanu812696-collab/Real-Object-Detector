# 🚀 Real-Time Object Detection using YOLOv8 & Streamlit

This project is a high-performance, web-based **Real-Time Object Detection System** built with Python. It leverages the state-of-the-art **YOLOv8** (You Only Look Once) model for lightning-fast object identification and **Streamlit** for a seamless user interface.

## 🌐 Live Demo
Check out the live application here: [https://real-object-detector.onrender.com](https://real-object-detector.onrender.com)
*(Note: Hosted on Render Free Tier, might take a few seconds to wake up).*

---

## ✨ Features
* **Real-Time Detection:** Processes live webcam feed with minimal latency.
* **YOLOv8 Integration:** Uses the 'Nano' version of YOLOv8 for optimized performance on cloud servers.
* **Cloud Deployed:** Fully functional on Render with automated CI/CD.
* **Responsive UI:** Simple and clean interface built with Streamlit.
* **Cross-Platform:** Works on any device with a browser and camera.

---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **AI Model:** Ultralytics YOLOv8
* **Frontend:** Streamlit
* **Computer Vision:** OpenCV
* **Deployment:** Render (Cloud PaaS)
* **Backend:** PyTorch

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/BHANU PRATAP SINGH / Real-Time Object Detection .git](https://real-object-detector.onrender.com)
   cd your-repo-name
   Create a Virtual Environment:

2. Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

3. Bash
pip install -r requirements.txt
Run the App:

4. Bash
streamlit run main.py

🔧 Configuration for Render
To ensure smooth deployment on Render, the following configurations were applied:

Requirements: Fixed torch==2.5.1 and torchvision to avoid security-related loading errors in newer versions.

Port Handling: Streamlit configured to run on Render's dynamic port.

Memory Optimization: Used yolov8n.pt to stay within the free tier RAM limits.
