"""Interactive Streamlit Web Application for Image Caption Generation with Spatial Attention Maps."""

import io
import sys
import time
from pathlib import Path
from PIL import Image

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import get_default_config
from src.inference.predictor import CaptionPredictor
from src.evaluation.metrics import evaluate_captions
from src.evaluation.visualization import plot_attention_heatmaps
from src.utils.device import get_device

# Page Configuration
st.set_page_config(
    page_title="Neural Image Captioning with Attention",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .caption-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #1e3c72;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    
    .caption-text {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a237e;
        line-height: 1.4;
    }
    
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1976D2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #616161;
        font-weight: 600;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Vision-Language Attention Model...")
def load_caption_predictor():
    """Caches and returns the CaptionPredictor singleton instance."""
    cfg = get_default_config()
    best_ckpt = cfg.paths.checkpoints_dir / "caption_model_best.pt"
    ckpt_path = best_ckpt if best_ckpt.exists() else None
    vocab_path = cfg.paths.vocab_path if cfg.paths.vocab_path.exists() else None

    predictor = CaptionPredictor(
        checkpoint_path=ckpt_path,
        vocab_path=vocab_path,
        config=cfg,
        device=get_device()
    )
    return predictor, cfg


def main():
    predictor, cfg = load_caption_predictor()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.2rem;">🖼️ Neural Image Caption Generator</h1>
        <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
            Deep Learning Vision-Language model with <b>ResNet-50 CNN + Bahdanau Spatial Attention + LSTM Decoder</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Generation Controls
    with st.sidebar:
        st.header("⚙️ Generation Parameters")
        
        decoding_method = st.radio(
            "Decoding Algorithm",
            options=["Beam Search", "Greedy Search"],
            index=0,
            help="Beam search explores multiple hypotheses for higher fluency. Greedy search picks the top token at each step."
        )

        method_key = "beam" if decoding_method == "Beam Search" else "greedy"

        if method_key == "beam":
            beam_width = st.slider("Beam Width (k)", min_value=1, max_value=10, value=5, step=1)
            temperature = 1.0
            rep_penalty = 1.0
        else:
            beam_width = 1
            temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
            rep_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

        max_length = st.slider("Maximum Length", min_value=10, max_value=50, value=30, step=5)

        st.markdown("---")
        st.subheader("Model Status")
        st.info(f"**Device:** `{predictor.device}`\n\n**Vocab Size:** `{len(predictor.vocab):,}` words\n\n**Backbone:** `{cfg.model.encoder_name}`")

    # Main Layout Tabs
    tab_generate, tab_eval, tab_arch = st.tabs(["✨ Generate Captions", "📊 Ground Truth Evaluation", "🧠 Architecture & Theory"])

    with tab_generate:
        col1, col2 = st.columns([1.1, 1.3], gap="large")

        with col1:
            st.subheader("1. Select or Upload Image")
            
            input_source = st.radio(
                "Image Source",
                options=["Sample Gallery", "Upload File", "Webcam Capture"],
                horizontal=True
            )

            selected_image = None
            sample_references = []

            if input_source == "Sample Gallery":
                sample_dir = cfg.paths.raw_data_dir / "Images"
                sample_files = list(sample_dir.glob("*.jpg"))[:6] if sample_dir.exists() else []

                if sample_files:
                    sample_cols = st.columns(3)
                    for idx, s_path in enumerate(sample_files):
                        with sample_cols[idx % 3]:
                            img_thumb = Image.open(s_path).resize((180, 140))
                            if st.button(f"Sample #{idx+1}", key=f"btn_sample_{idx}"):
                                st.session_state["active_image"] = s_path
                            st.image(img_thumb, use_container_width=True)
                    
                    active_path = st.session_state.get("active_image", sample_files[0])
                    selected_image = Image.open(active_path).convert("RGB")
                    st.image(selected_image, caption=f"Selected: {Path(active_path).name}", use_container_width=True)
                else:
                    st.warning("No sample gallery found. Upload an image below.")
                    uploaded_file = st.file_uploader("Upload Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
                    if uploaded_file:
                        selected_image = Image.open(uploaded_file).convert("RGB")
                        st.image(selected_image, caption="Uploaded Image", use_container_width=True)

            elif input_source == "Upload File":
                uploaded_file = st.file_uploader("Upload Image (JPEG, PNG, WebP)", type=["jpg", "jpeg", "png", "webp"])
                if uploaded_file:
                    selected_image = Image.open(uploaded_file).convert("RGB")
                    st.image(selected_image, caption="Uploaded Image", use_container_width=True)

            elif input_source == "Webcam Capture":
                camera_img = st.camera_input("Capture Photo")
                if camera_img:
                    selected_image = Image.open(camera_img).convert("RGB")
                    st.image(selected_image, caption="Captured Image", use_container_width=True)

        with col2:
            st.subheader("2. Generated Description & Attention")

            if selected_image is not None:
                if st.button("🚀 Generate Caption", type="primary", use_container_width=True):
                    with st.spinner("Analyzing visual features and generating sequence..."):
                        t0 = time.time()
                        result = predictor.predict(
                            image=selected_image,
                            method=method_key,
                            beam_width=beam_width,
                            max_len=max_length,
                            temperature=temperature,
                            repetition_penalty=rep_penalty
                        )
                        latency = (time.time() - t0) * 1000.0

                        st.session_state["last_result"] = result
                        st.session_state["last_latency"] = latency

                if "last_result" in st.session_state:
                    res = st.session_state["last_result"]
                    lat = st.session_state.get("last_latency", 0.0)

                    # Display Generated Caption Box
                    st.markdown(f"""
                    <div class="caption-box">
                        <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">
                            Generated Caption ({res.method.upper()})
                        </div>
                        <div class="caption-text">
                            "{res.caption}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Metrics stats
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(res.tokens)}</div><div class="metric-label">Words Generated</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-card"><div class="metric-val">{lat:.0f} ms</div><div class="metric-label">Inference Latency</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-card"><div class="metric-val">{res.confidence_score:.2f}</div><div class="metric-label">Confidence Score</div></div>', unsafe_allow_html=True)

                    st.markdown("---")
                    st.subheader("🔍 Visual Attention Heatmap Breakdown")
                    st.write("See which spatial region of the image the neural network attended to when generating each specific word:")

                    if res.attention_weights.size(0) > 0 and len(res.tokens) > 0:
                        fig = plot_attention_heatmaps(
                            image=selected_image,
                            words=res.tokens,
                            alphas=res.attention_weights,
                            smooth=True
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("No attention weights available for this sequence.")
            else:
                st.info("👈 Please select or upload an image to generate a caption.")

    with tab_eval:
        st.subheader("Evaluate Predictions Against Human References")
        st.write("Compare the generated caption against human-written ground truth reference captions (BLEU 1-4, ROUGE-L, METEOR).")

        cand_input = st.text_input(
            "Candidate Caption (Model Output)",
            value=st.session_state.get("last_result", None).caption if "last_result" in st.session_state else "a brown dog running across the grass"
        )

        st.write("**Human Reference Captions (Flickr8k 5-reference standard):**")
        r1 = st.text_input("Reference 1", value="a brown dog is running through the green grass")
        r2 = st.text_input("Reference 2", value="a cute dog playing on a lawn outside")
        r3 = st.text_input("Reference 3", value="a brown dog chasing something in the yard")
        r4 = st.text_input("Reference 4", value="a pet dog running in the park")
        r5 = st.text_input("Reference 5", value="a furry dog outdoors on the grass")

        if st.button("📊 Compute Evaluation Metrics"):
            refs = [r for r in [r1, r2, r3, r4, r5] if r.strip()]
            scores = evaluate_captions([cand_input], [refs])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("BLEU-1 (Unigram)", f"{scores['bleu_1']*100:.1f}%")
                st.metric("BLEU-2 (Bigram)", f"{scores['bleu_2']*100:.1f}%")
            with c2:
                st.metric("BLEU-3 (Trigram)", f"{scores['bleu_3']*100:.1f}%")
                st.metric("BLEU-4 (4-gram)", f"{scores['bleu_4']*100:.1f}%")
            with c3:
                st.metric("ROUGE-1 F1", f"{scores['rouge_1']*100:.1f}%")
                st.metric("ROUGE-2 F1", f"{scores['rouge_2']*100:.1f}%")
            with c4:
                st.metric("ROUGE-L F1", f"{scores['rouge_l']*100:.1f}%")
                st.metric("METEOR Score", f"{scores['meteor']*100:.1f}%")

    with tab_arch:
        st.subheader("System Architecture & Deep Learning Mechanics")
        st.markdown("""
        ### End-to-End Vision-Language Pipeline
        1. **Vision Encoder (Transfer Learning)**:
           - A pretrained CNN (**ResNet-50**) strips the final classification and average pooling layers, preserving the $14 \\times 14 \\times 2048$ spatial grid ($P = 196$ spatial patches).
        2. **Bahdanau Spatial Attention**:
           - Computes dynamic alignment energy between decoder hidden state $h_t$ and each spatial image patch $v_i$:
             $$\\alpha_{t, i} = \\frac{\\exp(e_{t, i})}{\\sum_{k=1}^P \\exp(e_{t, k})}, \\quad e_{t, i} = v_a^T \\tanh(W_{enc} v_i + W_{dec} h_t)$$
           - Produces context vector $z_t = \\sum_{i=1}^P \\alpha_{t, i} v_i$.
        3. **Adaptive Gating & LSTM Decoder**:
           - Gating scalar $\\beta_t = \\sigma(W_g h_t)$ controls the balance between visual features and language priors before feeding into the LSTM cell.
        4. **Doubly Stochastic Regularization**:
           - Ensures the model attends to every image region over the duration of the sentence:
             $$\\mathcal{L}_{reg} = \\lambda \\sum_{i=1}^P \\left(1 - \\sum_{t=1}^T \\alpha_{t, i}\\right)^2$$
        5. **Inference Algorithms**:
           - **Greedy Search**: $O(T)$ argmax token selection.
           - **Beam Search**: Maintains top-$k$ beam hypotheses with length penalty normalization $\\text{score} / \\left(\\frac{5 + L}{6}\\right)^\\alpha$.
        """)


if __name__ == "__main__":
    main()
