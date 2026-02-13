# 🩸 Fingerprint Blood Group Detection

This application uses a Convolutional Neural Network (CNN) to predict blood groups from fingerprint images using **Streamlit** and **TensorFlow**.

## 🚀 Demo
[Deploy on Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=reddycharan348/blood&branch=main&mainModule=app.py)

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/reddycharan348/blood.git
   cd blood
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📦 Deployment Note
This application uses **TensorFlow**, which is a large library (>500MB). 
**Vercel Serverless Functions** have a strict **250MB limit**, so deploying there will fail.

**✅ Recommended Deployment:** Use [Streamlit Community Cloud](https://streamlit.io/cloud) (Free & Unlimited size for public apps).

## 🤖 Model
The `bloodgroup_cnn_model.h5` file is a pre-trained CNN model expected to be in the root directory.
