"""
Streamlit App for ML Assignment 2
==================================
This app demonstrates all 6 classification models with evaluation metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E75B6;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #44546A;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND SCALER
# ============================================================================

@st.cache_resource
def load_models():
    """Load all saved models"""
    models = {}
    model_names = [
        'logistic_regression',
        'decision_tree',
        'k_nearest_neighbors',
        'naive_bayes',
        'random_forest',
        'xgboost'
    ]
    
    for name in model_names:
        try:
            with open(f'models/{name}.pkl', 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file {name}.pkl not found!")
            return None
    
    return models

@st.cache_resource
def load_scaler():
    """Load the saved scaler"""
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("Scaler not found. Using raw features.")
        return None

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # AUC Score
    try:
        if len(np.unique(y_true)) == 2:  # Binary
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:  # Multi-class
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                          multi_class='ovr', average='weighted')
    except:
        metrics['AUC'] = 0.0
    
    return metrics

def plot_confusion_matrix(cm, class_names):
    """Create an interactive confusion matrix plot"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500
    )
    
    return fig

def plot_metrics_radar(metrics):
    """Create radar chart for metrics"""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metrics',
        line_color='#2E75B6'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Performance Metrics Overview",
        height=400
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Machine Learning Classification Models</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### Dataset Upload")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Test Dataset (CSV)",
        type=['csv'],
        help="Upload your test dataset for predictions"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a test dataset (CSV file) to begin")
        st.markdown("""
        ### 📋 Instructions:
        1. **Upload your test dataset** using the sidebar
        2. **Select a model** from the dropdown
        3. **View predictions and metrics**
        4. **Analyze confusion matrix and classification report**
        
        ### 📊 Available Models:
        - Logistic Regression
        - Decision Tree Classifier
        - K-Nearest Neighbors
        - Naive Bayes (Gaussian)
        - Random Forest (Ensemble)
        - XGBoost (Ensemble)
        """)
        return
    
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return
    
    # Display dataset info
    with st.expander("📊 View Dataset Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.dataframe(df.head(10), use_container_width=True)
    
    # Model selection
    st.sidebar.markdown("### Model Selection")
    model_options = {
        'Logistic Regression': 'logistic_regression',
        'Decision Tree': 'decision_tree',
        'K-Nearest Neighbors': 'k_nearest_neighbors',
        'Naive Bayes': 'naive_bayes',
        'Random Forest': 'random_forest',
        'XGBoost': 'xgboost'
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Choose a model",
        list(model_options.keys())
    )
    
    # Load models
    models = load_models()
    scaler = load_scaler()
    
    if models is None:
        st.error("Failed to load models. Please ensure model files exist.")
        return
    
    selected_model_key = model_options[selected_model_name]
    model = models[selected_model_key]
    
    # Prepare data
    st.markdown(f'<h2 class="sub-header">🎯 Predictions using {selected_model_name}</h2>', 
                unsafe_allow_html=True)
    
    # Assume last column is target
    if df.shape[1] < 2:
        st.error("Dataset must have at least 2 columns (features + target)")
        return
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Scale if scaler exists
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    # Make predictions
    try:
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
        
        # Calculate metrics
        metrics = calculate_metrics(y, y_pred, y_pred_proba)
        
        # Display metrics
        st.markdown("### 📈 Evaluation Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        with col2:
            st.metric("AUC Score", f"{metrics['AUC']:.4f}")
        with col3:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col4:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col5:
            st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        with col6:
            st.metric("MCC", f"{metrics['MCC']:.4f}")
        
        # Visualizations
        st.markdown("---")
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("### 📊 Metrics Radar Chart")
            radar_fig = plot_metrics_radar(metrics)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col_viz2:
            st.markdown("### 🎯 Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            class_names = [str(c) for c in np.unique(y)]
            cm_fig = plot_confusion_matrix(cm, class_names)
            st.plotly_chart(cm_fig, use_container_width=True)
        
        # Classification Report
        st.markdown("### 📋 Detailed Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'), 
                    use_container_width=True)
        
        # Predictions DataFrame
        with st.expander("🔍 View Predictions", expanded=False):
            results_df = pd.DataFrame({
                'Actual': y.values,
                'Predicted': y_pred,
                'Correct': y.values == y_pred
            })
            st.dataframe(results_df, use_container_width=True)
            
            # Download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name=f'{selected_model_name}_predictions.csv',
                mime='text/csv'
            )
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 Machine Learning Assignment 2 | BITS Pilani | 2024</p>
        <p>Developed by [Your Name] | [Your BITS ID]</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
