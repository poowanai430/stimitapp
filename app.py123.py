import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="IrisStudio - Machine Learning",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom CSS for Yellow/Amber Theme & Modern Dark UI
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide top Streamlit header padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        padding: 18px 24px;
        border-radius: 16px;
        border: 1px solid #374151;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .logo-icon {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    
    .title-text {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .subtitle-text {
        font-size: 12px;
        color: #9ca3af;
        margin: 0;
    }
    
    /* Main Panel Styling */
    .panel-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .panel-title {
        font-size: 16px;
        font-weight: 600;
        color: #f3f4f6;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Result Display */
    .result-badge {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
    }
    
    .predicted-class {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin: 8px 0;
    }
    
    .description-text {
        color: #9ca3af;
        font-size: 13px;
        line-height: 1.5;
    }
    
    /* Custom Tip Box */
    .tip-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 12px;
        color: #d1d5db;
        margin-top: 15px;
    }
    
    /* Streamlit Slider Yellow Theme Customization */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #f59e0b !important;
    }
    div[data-baseweb="slider"] div {
        background-color: #f59e0b !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Mock Classification Algorithm (Softmax / Distance approximation)
# ---------------------------------------------------------
def predict_iris(sepal_l, sepal_w, petal_l, petal_w):
    # Centroids of Iris dataset (normalized approximation)
    centroids = {
        'Setosa': np.array([5.01, 3.43, 1.46, 0.25]),
        'Versicolor': np.array([5.94, 2.77, 4.26, 1.33]),
        'Virginica': np.array([6.59, 2.97, 5.55, 2.03])
    }
    
    features = np.array([sepal_l, sepal_w, petal_l, petal_w])
    
    # Calculate inverse Euclidean distances as logits
    distances = {species: np.linalg.norm(features - center) for species, center in centroids.items()}
    logits = np.array([-d * 2.5 for d in distances.values()])
    
    # Softmax probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    species_list = list(centroids.keys())
    pred_idx = np.argmax(probs)
    
    return species_list[pred_idx], probs[pred_idx], {
        'Setosa': probs[0],
        'Versicolor': probs[1],
        'Virginica': probs[2]
    }

# ---------------------------------------------------------
# Header Layout
# ---------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="logo-container">
        <div class="logo-icon">🌺</div>
        <div>
            <div class="title-text">IrisStudio</div>
            <div class="subtitle-text">Advanced Flower Classification & Analysis</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Navigation Tabs
tab_classifier, tab_guide, tab_dataset = st.tabs(["⚡ Classifier", "📖 Species Guide", "📊 Dataset Explorer"])

with tab_classifier:
    col_left, col_right = st.columns([1.1, 1.9], gap="medium")
    
    # ---------------------------------------------------------
    # LEFT COLUMN: Morphological Features Inputs
    # ---------------------------------------------------------
    with col_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎛️ Morphological Features <span style="font-size: 11px; color: #6b7280; margin-left: auto;">MEASUREMENTS IN CM</span></div>', unsafe_allow_html=True)
        
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8, step=0.1)
        sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0, step=0.1)
        petal_length = st.slider("Petal Length", 1.0, 7.0, 4.3, step=0.1)
        petal_width = st.slider("Petal Width", 0.1, 2.5, 1.3, step=0.1)
        
        st.markdown("""
        <div class="tip-box">
            💡 <strong>Tip:</strong> Petal dimensions are generally the strongest discriminators for distinguishing between Versicolor and Virginica species.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Perform Prediction
    # ---------------------------------------------------------
    pred_species, confidence, prob_dict = predict_iris(sepal_length, sepal_width, petal_length, petal_width)

    descriptions = {
        'Setosa': 'Characterized by distinctively smaller petal dimensions and wider sepal structures, usually found in cooler subarctic habitats.',
        'Versicolor': 'Characterized by medium petal and sepal dimensions, typically found in moist woodland habitats.',
        'Virginica': 'Characterized by robust and large petal structures with dark purple to violet patterns, native to eastern North America.'
    }

    # ---------------------------------------------------------
    # RIGHT COLUMN: Classification Result & Charts
    # ---------------------------------------------------------
    with col_right:
        # Top Result Box
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([2.5, 1])
        with res_col1:
            st.markdown('<span style="font-size: 11px; font-weight: 700; color: #f59e0b; letter-spacing: 1px;">CLASSIFICATION RESULT</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="predicted-class">Iris {pred_species} <span class="result-badge">{confidence*100:.1f}% Match</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="description-text">{descriptions[pred_species]}</div>', unsafe_allow_html=True)
        
        with res_col2:
            # Donut Confidence Gauge
            gauge_fig = go.Figure(go.Pie(
                values=[confidence, 1 - confidence],
                hole=0.75,
                sort=False,
                direction='clockwise',
                showlegend=False,
                hoverinfo='none',
                textinfo='none',
                marker=dict(colors=['#f59e0b', '#1f2937'])
            ))
            gauge_fig.add_annotation(
                text=f"<b>{int(confidence*100)}%</b><br><span style='font-size:10px;color:#9ca3af;'>CONFIDENCE</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color="#ffffff")
            )
            gauge_fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                height=130,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
            
        # Probability Bar Indicators
        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        prob_cols = st.columns(3)
        species_colors = {'Setosa': '#818cf8', 'Versicolor': '#f59e0b', 'Virginica': '#10b981'}
        
        for idx, (sp, p) in enumerate(prob_dict.items()):
            with prob_cols[idx]:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                    <span style="color: #d1d5db; font-weight: 600;">{sp}</span>
                    <span style="color: #9ca3af;">{p*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(p))

        st.markdown('</div>', unsafe_allow_html=True)

        # Bottom Row: Radar Chart & Donut Breakdown
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">🕸️ Feature Fingerprint</div>', unsafe_allow_html=True)
            
            # Normalize features for radar chart relative to max scales
            categories = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
            norm_vals = [
                sepal_length / 8.0,
                sepal_width / 4.5,
                petal_length / 7.0,
                petal_width / 2.5
            ]
            
            radar_fig = go.Figure(data=go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(245, 158, 11, 0.25)',
                line=dict(color='#f59e0b', width=2),
                marker=dict(size=6, color='#fbbf24')
            ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 1]),
                    angularaxis=dict(color='#9ca3af', font=dict(size=10)),
                    bgcolor='rgba(0,0,0,0)'
                ),
                margin=dict(t=20, b=20, l=30, r=30),
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(radar_fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_col2:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">🍩 Probability Weighting</div>', unsafe_allow_html=True)
            
            donut_fig = go.Figure(data=[go.Pie(
                labels=list(prob_dict.keys()),
                values=list(prob_dict.values()),
                hole=0.6,
                marker=dict(colors=['#818cf8', '#f59e0b', '#10b981']),
                textinfo='none',
                hoverinfo='label+percent'
            )])
            
            donut_fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='#9ca3af', size=11)
                )
            )
            st.plotly_chart(donut_fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Species Guide
# ---------------------------------------------------------
with tab_guide:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("### 🌸 Iris Species Overview")
    g_col1, g_col2, g_col3 = st.columns(3)
    
    with g_col1:
        st.markdown("#### Iris Setosa")
        st.caption("Short petals, wide sepals.")
        st.info("Avg Petal Length: **1.46 cm**\n\nAvg Petal Width: **0.24 cm**")
        
    with g_col2:
        st.markdown("#### Iris Versicolor")
        st.caption("Medium balance in proportions.")
        st.warning("Avg Petal Length: **4.26 cm**\n\nAvg Petal Width: **1.33 cm**")
        
    with g_col3:
        st.markdown("#### Iris Virginica")
        st.caption("Large leaves & distinct vibrant petals.")
        st.success("Avg Petal Length: **5.55 cm**\n\nAvg Petal Width: **2.03 cm**")
        
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 3: Dataset Explorer
# ---------------------------------------------------------
with tab_dataset:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Sample Dataset Preview")
    
    # Generate dummy Iris data for visualization
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'Sepal Length': np.random.uniform(4.3, 7.9, 30),
        'Sepal Width': np.random.uniform(2.0, 4.4, 30),
        'Petal Length': np.random.uniform(1.0, 6.9, 30),
        'Petal Width': np.random.uniform(0.1, 2.5, 30),
        'Species': np.random.choice(['Setosa', 'Versicolor', 'Virginica'], 30)
    })
    
    st.dataframe(sample_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)