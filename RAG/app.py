# app.py - Main Streamlit Application
import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="RAG Salary Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        margin: 1rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .data-info {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.extractedData = None
    st.session_state.textRep = None
    st.session_state.embeddings = None
    st.session_state.dataframe = None
    st.session_state.results_history = []
    st.session_state.api_key_configured = False

# Load environment variables
load_dotenv()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
    st.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("Enter Google API Key", type="password", 
                            help="Get your API key from Google AI Studio")
    if api_key:
        os.environ['API_KEY'] = api_key
        st.session_state.api_key_configured = True
    
    # File upload section
    st.subheader("📁 Data Source")
    
    # Option 1: Upload your own file
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], 
                                     help="Upload your salary dataset")
    
    if uploaded_file is not None:
        st.session_state.dataframe = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded: {uploaded_file.name}")
        st.info(f"📊 Shape: {st.session_state.dataframe.shape[0]} rows, {st.session_state.dataframe.shape[1]} columns")
    
    # Option 2: Use sample data
    st.markdown("### 📂 Or use sample data")
    if st.button("📊 Load Sample Data", use_container_width=True):
        try:
            # Try to load from data folder
            sample_path = "data/data_scientist_salaries.csv"
            if os.path.exists(sample_path):
                st.session_state.dataframe = pd.read_csv(sample_path)
                st.success(f"✅ Loaded sample data from {sample_path}!")
                st.info(f"📊 Shape: {st.session_state.dataframe.shape[0]} rows, {st.session_state.dataframe.shape[1]} columns")
            else:
                st.error(f"❌ Sample data file not found at {sample_path}")
                st.info("Please make sure the data folder contains 'data_scientist_salaries.csv'")
        except Exception as e:
            st.error(f"❌ Error loading sample: {str(e)}")
    
    # Show current data status
    if st.session_state.dataframe is not None:
        st.markdown("""
        <div class="data-info">
            <b>✅ Data loaded:</b><br>
            {} rows × {} columns
        </div>
        """.format(st.session_state.dataframe.shape[0], st.session_state.dataframe.shape[1]), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🤖 Model Settings")
    model_options = {
        "gemini-1.5-flash": "Gemini 1.5 Flash (Fast)",
        "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite (Balanced)",
        "gemini-1.5-pro": "Gemini 1.5 Pro (Powerful)"
    }
    selected_model = st.selectbox("Select Gemini Model", 
                                  options=list(model_options.keys()),
                                  format_func=lambda x: model_options[x])
    
    # Data preparation button
    if st.button("🚀 Initialize RAG System", use_container_width=True, type="primary"):
        if not st.session_state.api_key_configured:
            st.warning("⚠️ Please enter your Google API Key")
        elif st.session_state.dataframe is None:
            st.warning("⚠️ Please upload a CSV file or load sample data")
        else:
            with st.spinner("Initializing RAG system... This may take a minute..."):
                try:
                    # Configure API
                    genai.configure(api_key=api_key)
                    
                    # Important columns
                    impCols = ["Hobby", "OpenSource", "Country", "Student", "Employment", 
                              "FormalEducation", "UndergradMajor", "CompanySize", "DevType", 
                              "YearsCoding", "Salary", "SalaryType", "ConvertedSalary"]
                    
                    # Preprocessing function
                    def preprocessing(dataframe, impCols):
                        # Filter only available columns
                        available_cols = [col for col in impCols if col in dataframe.columns]
                        extractedData = dataframe[available_cols].copy()
                        
                        for col in available_cols:
                            if extractedData[col].isnull().sum() > 0:
                                if extractedData[col].dtype == 'object':
                                    extractedData[col] = extractedData[col].fillna('unknown')
                                    extractedData[col] = extractedData[col].astype(str).str.lower()
                                elif col == "ConvertedSalary":
                                    extractedData[col] = extractedData[col].fillna(extractedData[col].median())
                                else:
                                    extractedData[col] = extractedData[col].fillna(0)
                        
                        if 'ConvertedSalary' in extractedData.columns:
                            extractedData['ConvertedSalary'] = pd.to_numeric(
                                extractedData['ConvertedSalary'], errors='coerce'
                            )
                            extractedData['ConvertedSalary'] = extractedData['ConvertedSalary'].fillna(
                                extractedData['ConvertedSalary'].median()
                            )
                        
                        if 'YearsCoding' in extractedData.columns:
                            yearsexp = {
                                "0-2 years": 1, "3-5 years": 4, "6-8 years": 7, "9-11 years": 10,
                                "12-14 years": 13, "15-17 years": 16, "18-20 years": 19,
                                "21-23 years": 22, "24-26 years": 25, "27-29 years": 28, 
                                "30 or more years": 30
                            }
                            extractedData['YearsCodingNum'] = extractedData['YearsCoding'].map(yearsexp)
                            extractedData['YearsCodingNum'] = extractedData['YearsCodingNum'].fillna(
                                extractedData['YearsCodingNum'].median()
                            )
                        
                        return extractedData
                    
                    # Embedding function
                    def getEmbeddingsofCSV(extractedData, embeddingModel):
                        completeText = []
                        for _, rowData in extractedData.iterrows():
                            retrievedText = []
                            if 'ConvertedSalary' in rowData and rowData['ConvertedSalary'] > 0:
                                retrievedText.append(f"Converted Salary -> {str(rowData['ConvertedSalary'])}")
                            if 'YearsCoding' in rowData and str(rowData['YearsCoding']).strip() and rowData['YearsCoding'] != 'unknown':
                                retrievedText.append(f"Years Coding -> {str(rowData['YearsCoding'])}")
                                if 'YearsCodingNum' in rowData:
                                    retrievedText.append(f"Years Coding Num -> {str(rowData['YearsCodingNum'])}")
                            if 'Country' in rowData and rowData['Country'] is not None and rowData['Country'] != 'unknown':
                                retrievedText.append(f"Country -> {str(rowData['Country'])}")
                            if 'CompanySize' in rowData and str(rowData['CompanySize']).strip() and rowData['CompanySize'] != 'unknown':
                                retrievedText.append(f"Company Size -> {str(rowData['CompanySize'])}")
                            if 'DevType' in rowData and rowData['DevType'] is not None and rowData['DevType'] != 'unknown':
                                retrievedText.append(f"DevType -> {str(rowData['DevType'])}")
                            if 'FormalEducation' in rowData and rowData['FormalEducation'] is not None and rowData['FormalEducation'] != 'unknown':
                                retrievedText.append(f"FormalEducation -> {str(rowData['FormalEducation'])}")
                            
                            if retrievedText:  # Only add if there's data
                                completeText.append("; ".join(retrievedText))
                            else:
                                completeText.append("No data available")
                        
                        textembeddingsFromCSV = embeddingModel.encode(completeText)
                        return completeText, textembeddingsFromCSV
                    
                    # Initialize models
                    with st.spinner("Loading embedding model..."):
                        embeddingModel = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    with st.spinner("Preprocessing data..."):
                        st.session_state.extractedData = preprocessing(st.session_state.dataframe, impCols)
                    
                    with st.spinner("Generating embeddings..."):
                        st.session_state.textRep, st.session_state.embeddings = getEmbeddingsofCSV(
                            st.session_state.extractedData, embeddingModel
                        )
                    
                    st.session_state.initialized = True
                    
                    st.success("✅ RAG System initialized successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    
    # Reset button
    if st.button("🔄 Reset System", use_container_width=True):
        st.session_state.initialized = False
        st.session_state.extractedData = None
        st.session_state.textRep = None
        st.session_state.embeddings = None
        st.session_state.dataframe = None
        st.session_state.results_history = []
        st.rerun()
    
    # About section
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system helps you analyze salary data 
    using advanced AI. It supports both keyword and semantic search methods.
    
    **Features:**
    - 🔍 Keyword & Semantic Search
    - 📊 Data Visualization
    - 💬 AI-Powered Analysis
    - 📈 Salary Insights
    
    **Data Format Expected:**
    - CSV file with columns like DevType, YearsCoding, Country, ConvertedSalary, etc.
    """)

# Main content
st.markdown('<h1 class="main-header">💰 RAG Salary Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Salary Data Analysis with Retrieval-Augmented Generation</p>', unsafe_allow_html=True)

# Check if system is initialized
if st.session_state.initialized:
    
    # Data Overview Section
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(st.session_state.extractedData))
        with col2:
            st.metric("Total Columns", len(st.session_state.extractedData.columns))
        with col3:
            salary_col = st.session_state.extractedData['ConvertedSalary'] if 'ConvertedSalary' in st.session_state.extractedData.columns else pd.Series([0])
            avg_salary = salary_col[salary_col > 0].mean() if any(salary_col > 0) else 0
            st.metric("Avg Salary", f"${avg_salary:,.0f}")
        with col4:
            unique_roles = st.session_state.extractedData['DevType'].nunique() if 'DevType' in st.session_state.extractedData.columns else 0
            st.metric("Unique Roles", unique_roles)
        
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(st.session_state.extractedData.head(10), use_container_width=True)
        
        # Salary distribution
        if 'ConvertedSalary' in st.session_state.extractedData.columns:
            salary_data = st.session_state.extractedData[st.session_state.extractedData['ConvertedSalary'] > 0]
            if not salary_data.empty:
                fig = px.histogram(
                    salary_data,
                    x='ConvertedSalary',
                    title='Salary Distribution',
                    labels={'ConvertedSalary': 'Salary ($)', 'count': 'Frequency'},
                    nbins=50
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Query Input Section
    st.markdown("### 🔍 Ask a Question")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input(
            "Enter your question about the salary data",
            placeholder="e.g., What is the average salary of data scientists with over 5 years of experience?",
            label_visibility="collapsed"
        )
    with col2:
        retrieval_method = st.selectbox(
            "Retrieval Method",
            options=["semantic", "keyword"],
            format_func=lambda x: x.capitalize(),
            label_visibility="collapsed"
        )
    
    # Query button
    if st.button("🚀 Analyze", use_container_width=True, type="primary"):
        if user_question:
            with st.spinner("Analyzing your question..."):
                try:
                    # Download NLTK data if needed
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt', quiet=True)
                    try:
                        nltk.data.find('corpora/stopwords')
                    except LookupError:
                        nltk.download('stopwords', quiet=True)
                    
                    def keywordMatching(userQuery, textRep, extractedData):
                        eng_stopwords = set(stopwords.words('english'))
                        queryWords = []
                        
                        for word in userQuery.lower().split():
                            clean_word = word.strip(",.!?:;()[]{}")
                            if clean_word not in eng_stopwords and len(clean_word) > 2:
                                queryWords.append(clean_word)
                        
                        if not queryWords:
                            fallback_words = []
                            for word in userQuery.lower().split():
                                clean_word = word.strip(",.!?:;()[]{}")
                                if len(clean_word) > 2:
                                    fallback_words.append(clean_word)
                            queryWords = fallback_words
                        
                        keywordMatchScores = []
                        for index, rowData in enumerate(textRep):
                            scoreObtained = 0
                            for word in queryWords:
                                if word in rowData.lower():
                                    scoreObtained += 1
                            if scoreObtained > 0:
                                keywordMatchScores.append((scoreObtained, index))
                        
                        keywordMatchScores.sort(reverse=True)
                        
                        temp = []
                        for kmscore, index in keywordMatchScores:
                            if kmscore > 0:
                                temp.append(index)
                        indexesFinal = temp[:10]  # Limit to top 10
                        
                        if indexesFinal:
                            return extractedData.iloc[indexesFinal].copy()
                        else:
                            return extractedData.head(0)
                    
                    def semanticMatching(userQuery, embeddings, extractedData):
                        embeddingModel = SentenceTransformer('all-MiniLM-L6-v2')
                        embeddingofQuery = embeddingModel.encode([userQuery])
                        similarityScoreValues = cosine_similarity(embeddingofQuery, embeddings)[0]
                        indexesFinal = np.argsort(similarityScoreValues)[::-1][:10]  # Top 10
                        return extractedData.iloc[indexesFinal].copy()
                    
                    def retriever(userQuestion, method, textRep, extractedData, embeddings):
                        if method.lower() == 'keyword':
                            dataObtained = keywordMatching(userQuestion, textRep, extractedData)
                        else:  # semantic
                            dataObtained = semanticMatching(userQuestion, embeddings, extractedData)
                        
                        if dataObtained.empty:
                            return {
                                'question': userQuestion,
                                'method': method,
                                'data': dataObtained,
                                'response': "No relevant data found for your query. Please try a different question or method.",
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                        
                        contextList = []
                        for i, (_, rowVal) in enumerate(dataObtained.iterrows()):
                            contextList.append(f"\n**Entry {i+1}:**")
                            if 'DevType' in rowVal and rowVal['DevType'] != 'unknown':
                                contextList.append(f"- Role: {rowVal['DevType']}")
                            if 'YearsCoding' in rowVal and rowVal['YearsCoding'] != 'unknown':
                                contextList.append(f"- Experience: {rowVal['YearsCoding']}")
                            if 'Country' in rowVal and rowVal['Country'] != 'unknown':
                                contextList.append(f"- Country: {rowVal['Country']}")
                            if 'ConvertedSalary' in rowVal and rowVal['ConvertedSalary'] > 0:
                                contextList.append(f"- Salary: ${rowVal['ConvertedSalary']:,.0f}")
                            if 'CompanySize' in rowVal and rowVal['CompanySize'] != 'unknown':
                                contextList.append(f"- Company Size: {rowVal['CompanySize']}")
                            if 'FormalEducation' in rowVal and rowVal['FormalEducation'] != 'unknown':
                                contextList.append(f"- Education: {rowVal['FormalEducation']}")
                            if 'Employment' in rowVal and rowVal['Employment'] != 'unknown':
                                contextList.append(f"- Employment: {rowVal['Employment']}")
                        
                        context = "\n".join(contextList)
                        
                        prompt = f"""You are a data science analyst analyzing salary data. 
Based ONLY on the provided data context below, answer the question accurately.
If the question asks for averages or ranges, calculate them from the data.
If the data doesn't contain relevant information to answer the question fully, say so clearly.

DATA CONTEXT:
{context}

QUESTION: {userQuestion}

Provide a detailed, accurate answer based strictly on the data above. Include specific numbers and calculations where relevant."""
                        
                        model = genai.GenerativeModel(selected_model)
                        response = model.generate_content(prompt)
                        
                        return {
                            'question': userQuestion,
                            'method': method,
                            'data': dataObtained,
                            'response': response.text,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    
                    # Execute retrieval
                    result = retriever(
                        user_question,
                        retrieval_method,
                        st.session_state.textRep,
                        st.session_state.extractedData,
                        st.session_state.embeddings
                    )
                    
                    # Add to history
                    st.session_state.results_history.append(result)
                    
                    # Display results
                    st.markdown("### 📋 Analysis Results")
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="result-box">
                            <h4>📝 Question:</h4>
                            <p style="font-size: 1.1rem;">{result['question']}</p>
                            <h4>🔍 Retrieval Method:</h4>
                            <p><span style="background-color: #1E88E5; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-weight: 600;">{result['method'].upper()}</span></p>
                            <h4>💡 Answer:</h4>
                            <p style="font-size: 1rem; line-height: 1.6;">{result['response']}</p>
                            <p style="color: #666; font-size: 0.9rem; margin-top: 1rem;">⏱️ {result['timestamp']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show retrieved data
                    if not result['data'].empty:
                        with st.expander(f"📊 Retrieved Data ({len(result['data'])} entries)"):
                            st.dataframe(result['data'], use_container_width=True)
                            
                            if 'ConvertedSalary' in result['data'].columns:
                                salary_data = result['data'][result['data']['ConvertedSalary'] > 0]
                                if not salary_data.empty:
                                    fig = px.bar(
                                        salary_data.reset_index(),
                                        x=salary_data.index,
                                        y='ConvertedSalary',
                                        title='Salary Distribution in Retrieved Data',
                                        labels={'ConvertedSalary': 'Salary ($)', 'index': 'Entry'},
                                        color='ConvertedSalary',
                                        color_continuous_scale='Blues'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing question: {str(e)}")
                    st.exception(e)
    
    # Query History Section
    if st.session_state.results_history:
        st.markdown("### 📜 Query History")
        for i, result in enumerate(reversed(st.session_state.results_history[-5:])):
            with st.expander(f"Q{len(st.session_state.results_history)-i}: {result['question'][:70]}..."):
                st.markdown(f"**Method:** `{result['method'].upper()}`")
                st.markdown(f"**Time:** {result['timestamp']}")
                st.markdown(f"**Answer:** {result['response'][:200]}...")

else:
    # Welcome message when system not initialized
    st.markdown("""
    <div class="info-box" style="text-align: center; padding: 3rem;">
        <h2>👋 Welcome to RAG Salary Assistant!</h2>
        <p style="font-size: 1.2rem;">To get started, configure the system using the sidebar:</p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin: 2rem 0; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; background-color: #1E88E5; color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem;">1</div>
                <p style="font-weight: 600;">Enter your<br>Google API Key</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; background-color: #1E88E5; color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem;">2</div>
                <p style="font-weight: 600;">Upload CSV or<br>Load Sample Data</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; background-color: #1E88E5; color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem;">3</div>
                <p style="font-weight: 600;">Click "Initialize<br>RAG System"</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features showcase
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Smart Retrieval</h3>
            <p>Keyword and semantic search methods to find relevant data</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI Analysis</h3>
            <p>Powered by Google's Gemini models for accurate insights</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Visualizations</h3>
            <p>Interactive charts and data exploration</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        Made with ❤️ using Streamlit | RAG Salary Assistant | 
        <a href="https://github.com/yourusername/rag-salary-assistant" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)