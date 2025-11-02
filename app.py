import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import joblib
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .normal {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .pneumonia {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
CATEGORIES = ['NORMAL', 'PNEUMONIA']
MODEL_DIR = 'models'

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    try:
        # Load Deep Learning models
        if os.path.exists(f'{MODEL_DIR}/cnn_model.h5'):
            models['Custom CNN'] = load_model(f'{MODEL_DIR}/cnn_model.h5')
        if os.path.exists(f'{MODEL_DIR}/vgg16_model.h5'):
            models['VGG16'] = load_model(f'{MODEL_DIR}/vgg16_model.h5')
        if os.path.exists(f'{MODEL_DIR}/resnet50_model.h5'):
            models['ResNet50'] = load_model(f'{MODEL_DIR}/resnet50_model.h5')
        
        # Load Traditional ML models
        ml_models = ['logistic_regression', 'svm_model', 'random_forest', 
                     'decision_tree', 'naive_bayes']
        for model_name in ml_models:
            model_path = f'{MODEL_DIR}/{model_name}.pkl'
            if os.path.exists(model_path):
                models[model_name.replace('_', ' ').title()] = joblib.load(model_path)
        
        # Load PCA
        pca = None
        if os.path.exists(f'{MODEL_DIR}/pca_model.pkl'):
            pca = joblib.load(f'{MODEL_DIR}/pca_model.pkl')
        
        return models, pca
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, None

@st.cache_data
def load_model_results():
    """Load model comparison results"""
    try:
        if os.path.exists('model_comparison_results.csv'):
            df = pd.read_csv('model_comparison_results.csv', index_col=0)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

def preprocess_image_dl(image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocess image for deep learning models"""
    img = image.resize(target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_ml(image, pca, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocess image for traditional ML models"""
    img = image.convert('L')  # Convert to grayscale
    img = img.resize(target_size)
    img_array = np.array(img).flatten() / 255.0
    img_array = img_array.reshape(1, -1)
    if pca is not None:
        img_array = pca.transform(img_array)
    return img_array

def predict(image, model, model_name, pca=None):
    """Make prediction using the selected model"""
    try:
        # Check if it's a deep learning or traditional ML model
        is_dl_model = model_name in ['Custom CNN', 'VGG16', 'ResNet50']
        
        if is_dl_model:
            img_array = preprocess_image_dl(image)
            prediction = model.predict(img_array, verbose=0)[0][0]
        else:
            img_array = preprocess_image_ml(image, pca)
            prediction = model.predict_proba(img_array)[0][1]
        
        result = 'PNEUMONIA' if prediction > 0.5 else 'NORMAL'
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return result, confidence, prediction
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def display_prediction(result, confidence):
    """Display prediction result in a styled box"""
    if result == 'NORMAL':
        st.markdown(f"""
            <div class="prediction-box normal">
                ✅ Prediction: NORMAL<br>
                Confidence: {confidence:.2%}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box pneumonia">
                ⚠️ Prediction: PNEUMONIA<br>
                Confidence: {confidence:.2%}
            </div>
        """, unsafe_allow_html=True)
        st.warning("⚕️ This prediction suggests pneumonia. Please consult a healthcare professional for proper diagnosis.")

def create_confidence_gauge(confidence, result):
    """Create a gauge chart for confidence visualization"""
    color = '#28a745' if result == 'NORMAL' else '#dc3545'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': "Confidence Level", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffe6e6'},
                {'range': [50, 75], 'color': '#fff4e6'},
                {'range': [75, 100], 'color': '#e6ffe6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Pneumonia Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Chest X-Ray Analysis</p>', unsafe_allow_html=True)
    
    # Load models
    models, pca = load_models()
    results_df = load_model_results()
    
    if not models:
        st.error("⚠️ No models found! Please train the models first by running the Jupyter notebook.")
        st.info("📌 Run `pneumonia_detection_models.ipynb` to train all models.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/lungs.png", width=100)
        st.title("Settings")
        
        # Model selection
        st.subheader("Select Model")
        model_name = st.selectbox(
            "Choose a model for prediction:",
            options=list(models.keys()),
            index=0
        )
        
        # Information section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info("""
        This system uses machine learning to detect pneumonia from chest X-ray images.
        
        **Models Available:**
        - Deep Learning: CNN, VGG16, ResNet50
        - Traditional ML: SVM, Random Forest, etc.
        
        **Note:** This is for educational purposes only and should not replace professional medical diagnosis.
        """)
        
        # Dataset info
        if results_df is not None:
            st.markdown("---")
            st.subheader("📊 Model Performance")
            best_model = results_df['accuracy'].idxmax()
            best_acc = results_df['accuracy'].max()
            st.metric("Best Model", best_model)
            st.metric("Best Accuracy", f"{best_acc:.2%}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Model Comparison", "📖 Documentation"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Chest X-Ray Image")
            uploaded_file = st.file_uploader(
                "Choose an X-ray image (JPG, JPEG, PNG)",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a chest X-ray image for pneumonia detection"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded X-Ray", use_container_width=True)
                
                # Predict button
                if st.button("🔬 Analyze X-Ray", type="primary", use_container_width=True):
                    with st.spinner(f"Analyzing image with {model_name}..."):
                        result, confidence, raw_pred = predict(image, models[model_name], model_name, pca)
                        
                        if result is not None:
                            st.session_state.prediction_made = True
                            st.session_state.prediction_result = {
                                'result': result,
                                'confidence': confidence,
                                'raw_prediction': raw_pred,
                                'model_name': model_name
                            }
        
        with col2:
            if st.session_state.prediction_made and st.session_state.prediction_result:
                st.subheader("Analysis Results")
                
                pred_data = st.session_state.prediction_result
                
                # Display prediction
                display_prediction(pred_data['result'], pred_data['confidence'])
                
                # Display confidence gauge
                st.plotly_chart(
                    create_confidence_gauge(pred_data['confidence'], pred_data['result']),
                    use_container_width=True
                )
                
                # Additional metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Model Used", pred_data['model_name'])
                with col_b:
                    st.metric("Raw Score", f"{pred_data['raw_prediction']:.4f}")
                
                # Interpretation guide
                with st.expander("📋 Understanding the Results"):
                    st.markdown("""
                    **Confidence Level:**
                    - 🟢 **90-100%**: Very High Confidence
                    - 🟡 **75-89%**: High Confidence
                    - 🟠 **60-74%**: Moderate Confidence
                    - 🔴 **Below 60%**: Low Confidence
                    
                    **Important Notes:**
                    - This is a screening tool, not a diagnostic device
                    - Always consult a healthcare professional
                    - False positives and negatives can occur
                    - Image quality affects prediction accuracy
                    """)
    
    with tab2:
        st.subheader("📊 Model Performance Comparison")
        
        if results_df is not None:
            # Display results table
            st.dataframe(
                results_df.style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig_acc = px.bar(
                    results_df,
                    x=results_df.index,
                    y='accuracy',
                    title='Model Accuracy Comparison',
                    labels={'x': 'Model', 'accuracy': 'Accuracy'},
                    color='accuracy',
                    color_continuous_scale='Blues'
                )
                fig_acc.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                # F1-Score comparison
                fig_f1 = px.bar(
                    results_df,
                    x=results_df.index,
                    y='f1_score',
                    title='Model F1-Score Comparison',
                    labels={'x': 'Model', 'f1_score': 'F1-Score'},
                    color='f1_score',
                    color_continuous_scale='Greens'
                )
                fig_f1.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_f1, use_container_width=True)
            
            # Metrics comparison radar chart
            st.subheader("Multi-Metric Comparison")
            
            # Select models to compare
            models_to_compare = st.multiselect(
                "Select models to compare:",
                options=results_df.index.tolist(),
                default=results_df.index.tolist()[:3]
            )
            
            if models_to_compare:
                fig_radar = go.Figure()
                
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                
                for model in models_to_compare:
                    values = [results_df.loc[model, m] for m in metrics]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        fill='toself',
                        name=model
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Multi-Metric Model Comparison"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Model comparison results not found. Please train the models first.")
    
    with tab3:
        st.subheader("📖 Project Documentation")
        
        st.markdown("""
        ### Pneumonia Detection from Chest X-Rays
        
        #### Overview
        This project implements multiple machine learning and deep learning models to detect pneumonia from chest X-ray images.
        
        #### Dataset
        - **Source**: Kaggle Chest X-Ray Pneumonia Dataset
        - **Classes**: Normal and Pneumonia
        - **Training Images**: ~5,200 images
        - **Test Images**: ~600 images
        
        #### Models Implemented
        
        **Traditional Machine Learning:**
        1. Logistic Regression
        2. Support Vector Machine (SVM)
        3. Random Forest
        4. Decision Tree
        5. Naive Bayes
        
        **Deep Learning:**
        1. Custom CNN - Custom convolutional neural network
        2. VGG16 - Transfer learning with VGG16
        3. ResNet50 - Transfer learning with ResNet50
        
        #### Methodology
        
        1. **Data Preprocessing**
           - Image resizing to 150x150 pixels
           - Normalization (0-1 scaling)
           - Data augmentation for training
        
        2. **Feature Extraction**
           - PCA for dimensionality reduction (traditional ML)
           - Convolutional layers (deep learning)
        
        3. **Training**
           - Early stopping to prevent overfitting
           - Learning rate reduction on plateau
           - Model checkpointing
        
        4. **Evaluation Metrics**
           - Accuracy
           - Precision
           - Recall
           - F1-Score
        
        #### How to Use
        
        1. Upload a chest X-ray image
        2. Select a model from the sidebar
        3. Click "Analyze X-Ray"
        4. View the prediction and confidence score
        
        #### Important Notes
        
        ⚠️ **Disclaimer**: This is an educational project and should NOT be used for actual medical diagnosis. 
        Always consult qualified healthcare professionals for medical advice.
        
        #### Technical Stack
        - **Frontend**: Streamlit
        - **Backend**: TensorFlow/Keras, Scikit-learn
        - **Visualization**: Plotly, Matplotlib
        - **Image Processing**: OpenCV, PIL
        
        #### Performance
        Deep learning models (CNN, VGG16, ResNet50) generally achieve higher accuracy (85-95%) 
        compared to traditional ML models (70-85%) on this task.
        """)
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Developer**: ML Project Team")
        with col2:
            st.markdown("**Version**: 1.0.0")
        with col3:
            st.markdown("**License**: Educational Use")

if __name__ == "__main__":
    main()
