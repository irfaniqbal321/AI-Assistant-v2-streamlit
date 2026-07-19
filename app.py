import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Page Config
st.set_page_config(page_title="AI Data Assistant", layout="wide", page_icon="🤖")
load_dotenv()

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
       font-size: 3rem;
       font-weight: 900;
       color: #00D4FF;
       text-align: center;
       margin-bottom: 0.3rem;
       letter-spacing: 2px;
       text-shadow: 0 0 15px #00D4FF;
   }
.icon-header {
       font-size: 3.5rem;
       vertical-align: middle;
       margin-right: 15px;
       filter: drop-shadow(0 0 15px #00D4FF);
   }
.sub-header {
       font-size: 1.2rem;
       color: #B0B0B0;
       text-align: center;
       margin-bottom: 3rem;
       font-weight: 500;
   }
.stButton>button {
       width: 100%;
       background-color: #4F8BF9;
       color: white;
       border-radius: 10px;
       font-weight: 600;
       border: none;
       padding: 0.6rem;
   }
.stButton>button:hover {
       background-color: #3A7BE0;
   }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header"><span class="icon-header">🤖</span> AI Data Analysis Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload CSV • Ask Questions • Generate Charts • Get AI Explanation</p>', unsafe_allow_html=True)

# Configure Gemini API with System Instruction - FINAL FIX
@st.cache_resource
def setup_gemini():
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            'gemini-3.5-flash', # FINAL MODEL NAME
            system_instruction="""Tum ek Professional AI Data Analyst Assistant ho. Naam: DataBot.
            Sakht Rules:
            1. Hamesha saaf, formal aur professional roman Urdu me jawab do.
            2. "Jani", "yaar", "bhai", "dekho", emoji aur slang ka use sakhti se mana hai.
            3. Jawab seedha, concise aur 3-4 lines me do.
            4. Data ki baat karte waqt professional aur informative raho."""
        )
        return model, True
    except Exception as e:
        st.sidebar.error(f"API Key error: {e}")
        return None, False

model, api_available = setup_gemini()
if api_available:
    st.sidebar.success("✅ AI Model: gemini-2.0-flash Connected")

# Sidebar
with st.sidebar:
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader("CSV file upload karo", type=["csv"])
    st.markdown("---")
    st.info("Tip: CSV me 1 text aur 1 numeric column ho to chart best banega")

# Main Logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # TABS for clean UI
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "❓ Q&A", "📈 Charts", "🧠 AI Insights"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

    with tab2:
        st.subheader("Ask Questions")
        question = st.selectbox("Select a question:", [
            "Show basic statistics",
            "Show column names and data types",
            "Find missing values",
            "Show correlation between numeric columns"
        ])
        if st.button("Get Answer"):
            if question == "Show basic statistics":
                st.write(df.describe())
                answer = "Ye table me har numeric column ka mean, min, max aur count hai."
            elif question == "Show column names and data types":
                st.write(df.dtypes)
                answer = "Ye har column ka naam aur uska data type hai."
            elif question == "Find missing values":
                st.write(df.isnull().sum())
                answer = "Ye batata hai har column me kitne khali cells hain."
            elif question == "Show correlation between numeric columns":
                st.write(df.corr(numeric_only=True))
                answer = "Ye batata hai kaunse columns ek dusre se related hain."
            st.success(f"💡 Explanation: {answer}")

    with tab3:
        st.subheader("📊 Auto Chart Generator")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            col1, col2 = st.columns(2)
            with col1: x_axis = st.selectbox("X-axis - Category", categorical_cols)
            with col2: y_axis = st.selectbox("Y-axis - Number", numeric_cols)
            chart_type = st.radio("Chart Type", ["Bar Chart", "Line Chart"], horizontal=True)

            if st.button("Generate Chart"):
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.set_style("whitegrid")
                if chart_type == "Bar Chart":
                    sns.barplot(data=df, x=x_axis, y=y_axis, estimator='mean', ax=ax, hue=x_axis, palette="Blues_d", legend=False)
                else:
                    sns.lineplot(data=df, x=x_axis, y=y_axis, estimator='mean', ax=ax, marker='o')

                plt.title(f"Average {y_axis} by {x_axis}", fontsize=14)
                plt.xlabel(x_axis); plt.ylabel(f"Average {y_axis}"); plt.xticks(rotation=45)
                st.pyplot(fig)
                st.session_state['last_chart_data'] = (df, x_axis, y_axis) # Save for AI tab
        else:
            st.warning("⚠️ Chart ke liye 1 numeric aur 1 text column chahiye")

    with tab4:
        st.subheader("🧠 AI Result Explanation")
        if api_available and 'last_chart_data' in st.session_state:
            df_ai, x_col, y_col = st.session_state['last_chart_data']
            if st.button("Explain This Chart with AI"):
                try:
                    data = df_ai.groupby(x_col)[y_col].mean().round(2).to_dict()
                    prompt = f"""
                    Tum ek professional Data Analyst ho.
                    Chart: Average of '{y_col}' grouped by '{x_col}'
                    Data: {data}

                    Task: Is chart ko 3 lines me explain karo saaf Urdu me.
                    Koi slang, "jani", "yaar" use mat karo.
                    Pehle overall trend batao, phir sabse zyada aur sabse kam value ka zikr karo.
                    """
                    with st.spinner("AI is analyzing..."):
                        response = model.generate_content(prompt)
                        st.success("**AI Explanation:**")
                        st.write(response.text)
                except Exception as e:
                    st.error(f"AI se jawab nahi mila: {e}")
        elif not api_available:
            st.error("API Key missing hai..env file check karo")
        else:
            st.info("Pehle 'Charts' tab me chart generate karo, phir AI explanation aayegi")

else:
    st.info("👆 Sidebar se CSV file upload karo shuru karne ke liye")