import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load.env file
load_dotenv()

st.set_page_config(page_title="AI Data Assistant", layout="wide")
st.title("🤖 AI Data Analysis Assistant")

# Configure Gemini API - FORCE WORKING MODEL
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Naye users ke liye ye 2 models pakka chalte hain
    model = genai.GenerativeModel('gemini-3.5-flash')

    api_available = True
    st.sidebar.success("Using Model: gemini-3.5-flash")

except Exception as e:
    api_available = False
    st.error(f"API Key error: {e}")
    st.info("Tip:.env file me key check karo")

# Function for Step 5: AI Explanation
def generate_ai_explanation(df, x_col, y_col):
    if not api_available:
        st.error("API Key missing hai")
        return

    try:
        data = df.groupby(x_col)[y_col].mean().round(2).to_dict()
        prompt = f"""
        You are a data analyst.
        Chart: Average of '{y_col}' grouped by '{x_col}'
        Data: {data}
        Task: Write 2 simple lines explaining what the chart shows. Use Urdu-English mix. No jargon.
        """

        with st.spinner("AI is thinking..."):
            response = model.generate_content(prompt)
            st.success("**AI Explanation:** " + response.text)
    except Exception as e:
        st.error(f"AI se jawab nahi mila: {e}")

# 1. CSV Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.subheader("Dataset Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # 2. Predefined Questions
    st.subheader("Ask Questions")
    question = st.selectbox("Select a question:", [
        "Show basic statistics","Show column names and data types",
        "Find missing values","Show correlation between numeric columns"
    ])
    if st.button("Get Answer"):
        if question == "Show basic statistics": st.write(df.describe()); answer = "Ye table me har numeric column ka mean, min, max aur count hai."
        elif question == "Show column names and data types": st.write(df.dtypes); answer = "Ye har column ka naam aur uska data type hai."
        elif question == "Find missing values": st.write(df.isnull().sum()); answer = "Ye batata hai har column me kitne khali cells hain."
        elif question == "Show correlation between numeric columns": st.write(df.corr(numeric_only=True)); answer = "Ye batata hai kaunse columns ek dusre se related hain."
        st.success(f"Explanation: {answer}")

    # 3. Generate Chart
    st.subheader("📊 Auto Chart")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        col1, col2 = st.columns(2)
        with col1: x_axis = st.selectbox("X-axis - Category", categorical_cols)
        with col2: y_axis = st.selectbox("Y-axis - Number", numeric_cols)

        if st.button("Generate Chart"):
            fig, ax = plt.subplots(figsize=(8, 5))
            df.groupby(x_axis)[y_axis].mean().plot(kind='bar', ax=ax, color='skyblue')
            plt.title(f"Average {y_axis} by {x_axis}"); plt.xlabel(x_axis); plt.ylabel(f"Average {y_axis}"); plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("Step 5: AI Result Explanation")
            generate_ai_explanation(df, x_axis, y_axis)
    else:
        st.warning("Chart ke liye 1 numeric aur 1 text column chahiye")

else:
    st.info("Please upload a CSV file to start")