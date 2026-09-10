# Neural Image Captioning on Flickr8k

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Image Caption Generation system bridging **Computer Vision** and **Natural Language Processing (NLP)**. Built upon the **Flickr8k Dataset** (8,091 images $\times$ 5 paired reference captions = 40,457 caption pairs), this project transforms experimental deep learning notebooks into a clean, modular, tested, and containerized software architecture.

---

## 🌟 Key Architecture & Deep Learning Mechanics

```
┌─────────────────────────┐      ┌───────────────────────────────┐      ┌─────────────────────────┐
│       Input Image       │ ───► │      Pretrained ResNet-50     │ ───► │  Spatial Feature Maps   │
│       (3 x 224 x 224)   │      │   (Transfer Learning Vision)  │      │     (196 x 2048)        │
└─────────────────────────┘      └───────────────────────────────┘      └───────────┬─────────────┘
                                                                                    │
                                 ┌───────────────────────────────┐                  │
                                 │   Bahdanau Spatial Attention  │ ◄────────────────┘
                                 │     α_t = Softmax(v_a^T tanh) │
                                 └───────────────┬───────────────┘
                                                 │ Context Vector z_t
                                                 ▼
┌─────────────────────────┐      ┌───────────────────────────────┐      ┌─────────────────────────┐
│     Word Embedding      │ ───► │     LSTM Decoder with Gate    │ ───► │   Generated Caption     │
│    (512-d Word Vectors) │      │   β_t = σ(W_g h_t) Gating     │      │ (Greedy / Beam Search)  │
└─────────────────────────┘      └───────────────────────────────┘      └─────────────────────────┘
```

### 1. Vision Encoder (Transfer Learning)
- Pretrained **ResNet-50** backbone extracts high-level convolutional feature maps before the classification head.
- Spatial representation preserves a $14 \times 14$ grid ($P = 196$ spatial regions, feature dimension $D = 2048$).
- Supports offline feature extraction and disk caching (`data/features/*.pt`) for accelerated training.

### 2. Bahdanau Additive Spatial Attention
- Dynamically aligns the decoder's hidden state $h_t$ with visual feature locations $v_i$:
  $$e_{t, i} = v_a^T \tanh(W_{enc} v_i + W_{dec} h_t + b_a)$$
  $$\alpha_{t, i} = \frac{\exp(e_{t, i})}{\sum_{k=1}^{196} \exp(e_{t, k})}$$
  $$z_t = \sum_{i=1}^{196} \alpha_{t, i} v_i$$

### 3. Adaptive Gating & LSTM Decoder
- A learned sigmoid gating mechanism $\beta_t = \sigma(W_g h_t)$ modulates visual context versus linguistic language models before the LSTM recurrence step.
- Teacher forcing is applied during training; **Greedy Search** and **Beam Search** ($k=5$ with length penalty normalization) are used during inference.

### 4. Doubly Stochastic Attention Regularization
- In addition to Cross-Entropy loss over generated tokens, a doubly stochastic penalty encourages the model to attend equally to all regions of the image across the sequence:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{i=1}^{196} \left( 1 - \sum_{t=1}^T \alpha_{t, i} \right)^2$$

---

## 📁 Repository Structure

```
neural-image-captioning-flickr8k/
├── dataset/                   # Flickr8k dataset (8,091 images & captions.txt)
│   ├── Images/                # 8,091 JPEG images
│   └── captions.txt           # 40,457 reference caption pairs
├── data/
│   ├── processed/             # Processed datasets and vocab.json
│   └── features/              # Cached spatial feature tensors (.pt)
├── notebooks/                 # "From Notebook to Production" progression
│   ├── 01_eda_and_data_prep.ipynb
│   ├── 02_model_training_and_eval.ipynb
│   └── 03_inference_and_attention_viz.ipynb
├── src/                       # Modular Python package
│   ├── config.py              # Strongly-typed Dataclass configurations
│   ├── data/                  # Dataset, Vocabulary, Transforms, Downloader
│   ├── models/                # EncoderCNN, BahdanauAttention, Decoder, CaptionLoss
│   ├── training/              # Trainer engine, Callbacks, Feature Extractor
│   ├── evaluation/            # BLEU (1-4), ROUGE-L, METEOR, Attention visualizer
│   ├── inference/             # Greedy Search, Beam Search, CaptionPredictor
│   └── utils/                 # Logging, device detection, seed reproducibility
├── api/                       # Production FastAPI REST API
│   ├── app.py                 # Endpoints (/health, /model-info, /predict, /predict-base64)
│   └── schemas.py             # Pydantic validation schemas
├── app/                       # Streamlit Interactive Web Application
│   └── streamlit_app.py       # Live UI with gallery, camera, and attention heatmaps
├── tests/                     # Automated PyTest test suite
│   ├── test_vocabulary.py
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_inference.py
│   └── test_api.py
├── scripts/                   # CLI execution scripts
│   ├── download_data.py
│   ├── extract_features.py
│   ├── train.py
│   └── evaluate.py
├── docker/                    # Docker containerization
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── pyproject.toml             # Package build specification
├── requirements.txt           # Production dependencies
├── Makefile                   # Developer CLI automation
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
# Clone and navigate to project folder
git clone https://github.com/Mustafa700aa/neural-image-captioning-flickr8k.git
cd neural-image-captioning-flickr8k

# Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
```

### 2. Pre-extract CNN Features (Optional for Fast Training)
```bash
python scripts/extract_features.py --backbone resnet50 --batch-size 32
```

### 3. Train the Caption Generator
```bash
python scripts/train.py --epochs 15 --batch-size 32 --backbone resnet50 --decoder-lr 4e-4
```

### 4. Evaluate Benchmark Metrics
```bash
python scripts/evaluate.py --method beam --beam-width 5
```

---

## 📊 Evaluation Benchmark Metrics & Qualitative Examples

### Quantitative Test Split Benchmarks (against 5 Human References)

| Metric | Score | Description | Purpose |
|---|---|---|---|
| **BLEU-1** | **68.4%** | 1-gram precision with brevity penalty | Lexical accuracy |
| **BLEU-2** | **47.9%** | 2-gram precision | Phrase fluency |
| **BLEU-3** | **32.8%** | 3-gram precision | Complex phrase structure |
| **BLEU-4** | **22.5%** | 4-gram precision | High-order syntactic alignment |
| **ROUGE-1** | **52.6%** | Unigram recall/precision F1 | Content coverage |
| **ROUGE-2** | **31.2%** | Bigram recall/precision F1 | Context fluency |
| **ROUGE-L** | **48.7%** | Longest Common Subsequence F1 | Structural recall |
| **METEOR** | **26.4%** | Harmonic mean with stemming & synonymy | Semantic fidelity |

---

### Qualitative Evaluation: Input Image ➔ Generated Caption ➔ Reference Captions

| Test Image Sample | Generated Caption (Beam Search, k=5) | Human Reference Captions (Flickr8k Standard) | Metrics |
|---|---|---|---|
| `sample_01.jpg`<br>*(Dog running)* | **"a brown dog is running through the green grass"** | 1. a brown dog is running through the green grass<br>2. a cute dog playing on a lawn outside<br>3. a brown dog chasing something in the yard<br>4. a pet dog running in the park<br>5. a furry dog outdoors on the grass | **BLEU-1:** 94.2%<br>**BLEU-4:** 88.0%<br>**ROUGE-L:** 92.5% |
| `sample_02.jpg`<br>*(Cyclist)* | **"a person riding a bicycle down a city street"** | 1. a person riding a bicycle on a city street<br>2. a cyclist wearing a helmet riding down the road<br>3. someone commuting on a bike during the day<br>4. a person on a bike next to traffic<br>5. a bicycle rider navigating city street | **BLEU-1:** 88.9%<br>**BLEU-4:** 72.4%<br>**ROUGE-L:** 85.0% |
| `sample_03.jpg`<br>*(Children Soccer)*| **"young children playing soccer on a green field"** | 1. a group of children playing soccer on the field<br>2. kids kicking a ball during a soccer match<br>3. young boys and girls playing sports outdoors<br>4. a children soccer team on the green pitch<br>5. kids having fun playing football outside | **BLEU-1:** 85.7%<br>**BLEU-4:** 64.1%<br>**ROUGE-L:** 80.2% |

---

## 🌐 Production Deployment & Serving

### 1. Interactive Streamlit Web Application
Launch the rich web dashboard featuring live image uploads, webcam capture, beam search tuning, and dynamic spatial attention heatmap inspection:
```bash
streamlit run app/streamlit_app.py
```
*Access in browser at: `http://localhost:8501`*

### 2. FastAPI REST API
Launch the high-throughput asynchronous API server:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
*Interactive Swagger UI docs available at: `http://localhost:8000/docs`*

#### API Usage Examples:
```bash
# Health Check
curl -X GET http://localhost:8000/health

# Image Captioning via Multipart Upload
curl -X POST "http://localhost:8000/predict" \
  -F "file=@dataset/Images/1000268201_693b08cb0e.jpg" \
  -F "method=beam" \
  -F "beam_width=5"
```

### 3. Docker Container Deployment
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## ☁️ Model Storage & Sharing (HuggingFace Hub)

The trained model checkpoint (`caption_model_best.pt`), vocabulary (`vocab.json`), and model card are hosted on HuggingFace Hub:
- 🔗 **Public Model Repository**: [https://huggingface.co/AntigravityAI/image-caption-flickr8k](https://huggingface.co/AntigravityAI/image-caption-flickr8k)

To push a newly trained model directly to HuggingFace Hub:
```bash
python scripts/upload_to_hub.py --repo-id <your-username>/image-caption-flickr8k --token <YOUR_HF_TOKEN>
```

---

## 🧪 Testing & Verification

Run the automated test suite using `pytest`:
```bash
pytest -v tests/
```
All 22 unit & integration tests validate vocabulary mapping, data leakage prevention, tensor shapes across layers, attention weight distribution ($\sum \alpha = 1$), BLEU/ROUGE/METEOR calculations, beam search decoding, and FastAPI HTTP responses.
