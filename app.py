import json
import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="FashionForge AI | Generative Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "FashionForgeAI_Generator.h5"
MAPPING_PATH = ROOT / "models" / "category_mapping.json"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --bg:#0b0b10; --panel:#15151e; --muted:#aaa9bb; --accent:#c4a7ff; }
.stApp { background: radial-gradient(ellipse at 85% 0%, #29203d 0%, #11111a 34%, #0b0b10 72%); color:#f7f4ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1400px; }
.hero { padding:1.6rem 0 1rem 0; }
.eyebrow { color:#c4a7ff; text-transform:uppercase; letter-spacing:.22em; font-size:.76rem; font-weight:700; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.6rem,5vw,4.7rem); line-height:1.0; font-weight:700; margin:.65rem 0 1rem; }
.hero-title span { color:#c4a7ff; }
.hero-copy { max-width:720px; color:#b9b7c9; font-size:1.05rem; line-height:1.75; }
.pill { display:inline-block; border:1px solid #403650; background:#211b2d; color:#e0d2ff; padding:.4rem .75rem; border-radius:999px; font-size:.78rem; margin:0 .35rem .3rem 0; }
.panel { background:rgba(23,22,32,.82); border:1px solid #302b3b; border-radius:20px; padding:1.15rem 1.25rem; }
.metric { color:#a9a6b9; font-size:.8rem; }
.metric-value { color:#f8f5ff; font-size:1.45rem; font-weight:700; font-family:'Space Grotesk',sans-serif; }
div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#a78bfa,#d8b4fe); color:#15111f; border:0; border-radius:12px; font-weight:700; min-height:3rem; }
div.stButton > button { border-radius:10px; }
section[data-testid="stSidebar"] { background:#111019; border-right:1px solid #302b3b; }
[data-testid="stImage"] img { border-radius:14px; border:1px solid #302b3b; }
hr { border-color:#302b3b; }
.small-muted { color:#aaa9bb; font-size:.88rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_generator(path_string: str):
    # The supplied file is a trusted Keras HDF5 model. compile=False avoids
    # requiring training losses/optimizers at inference time.
    return tf.keras.models.load_model(path_string, compile=False, safe_mode=False)

@st.cache_data
def load_mapping(path_string: str):
    return json.loads(Path(path_string).read_text(encoding="utf-8"))

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered creative lab</div>
  <div class="hero-title">Fashion, <span>reimagined.</span></div>
  <div class="hero-copy">Explore fresh visual concepts with FashionForge AI — a conditional generative model trained to create experimental clothing designs by category.</div>
  <div style="margin-top:1rem">
    <span class="pill">Conditional GAN</span><span class="pill">128 × 128 generation</span><span class="pill">Downloadable concepts</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists() or not MAPPING_PATH.exists():
    st.error("Model files are missing. Keep the .h5 generator and category_mapping.json inside the models/ folder.")
    st.stop()

try:
    metadata = load_mapping(str(MAPPING_PATH))
    generator = load_generator(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the generator. Check TensorFlow compatibility and the model file. Details: {exc}")
    st.stop()

class_names = metadata["class_names"]
class_to_id = metadata["class_to_id"]
latent_dim = int(metadata.get("latent_dim", 128))
image_size = int(metadata.get("image_size", 128))

with st.sidebar:
    st.markdown("### ✦ Studio controls")
    st.markdown('<div class="small-muted">Shape a batch of concepts for your next collection.</div>', unsafe_allow_html=True)
    category = st.selectbox("Clothing category", class_names)
    count = st.slider("Number of concepts", 1, 8, 4)
    seed = st.number_input("Variation seed", min_value=0, max_value=999999, value=42, step=1)
    st.markdown("---")
    st.markdown("**Generation settings**")
    st.caption(f"Resolution: {image_size} × {image_size}px")
    st.caption(f"Latent vector: {latent_dim} dimensions")
    generate = st.button("✦ Generate collection", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Tip: change the seed to explore different variations in the same category.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="panel"><div class="metric">MODEL</div><div class="metric-value">FashionForge</div><div class="small-muted">Conditional generator</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="panel"><div class="metric">CATEGORIES</div><div class="metric-value">{len(class_names)}</div><div class="small-muted">Learned label embeddings</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="panel"><div class="metric">OUTPUT</div><div class="metric-value">PNG + ZIP</div><div class="small-muted">Save concepts locally</div></div>', unsafe_allow_html=True)

if "generated_batch" not in st.session_state:
    st.session_state.generated_batch = None
    st.session_state.generated_meta = None

if generate:
    category_id = int(class_to_id[category])
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0, 1, size=(count, latent_dim)).astype(np.float32)
    labels = np.full((count, 1), category_id, dtype=np.int32)

    with st.spinner("Composing your collection…"):
        output = generator([noise, labels], training=False).numpy()
        output = np.clip((output + 1.0) * 127.5, 0, 255).astype(np.uint8)

    st.session_state.generated_batch = output
    st.session_state.generated_meta = {"category": category, "seed": int(seed)}

if st.session_state.generated_batch is not None:
    batch = st.session_state.generated_batch
    meta = st.session_state.generated_meta
    st.markdown("---")
    st.markdown(f"## Your {meta['category']} collection")
    st.markdown(f'<div class="small-muted">Generated with variation seed {meta["seed"]}. Images are experimental model outputs, not production-ready garment specifications.</div>', unsafe_allow_html=True)

    grid = st.columns(4)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, arr in enumerate(batch):
            image = Image.fromarray(arr)
            file_buffer = io.BytesIO()
            image.save(file_buffer, format="PNG")
            data = file_buffer.getvalue()
            filename = f"fashionforge_{meta['category'].lower()}_{i+1:02d}.png"
            archive.writestr(filename, data)
            with grid[i % 4]:
                st.image(image, caption=f"Concept {i+1}", use_container_width=True)
                st.download_button(
                    "Download PNG",
                    data=data,
                    file_name=filename,
                    mime="image/png",
                    key=f"png_{meta['seed']}_{meta['category']}_{i}"
                )

    st.download_button(
        "Download all concepts (.zip)",
        data=zip_buffer.getvalue(),
        file_name=f"fashionforge_{meta['category'].lower()}_collection.zip",
        mime="application/zip",
        type="primary",
    )
else:
    st.markdown("---")
    st.markdown("## Your creative canvas")
    st.markdown('<div class="panel"><h3>Ready when you are.</h3><div class="small-muted">Choose a category in the sidebar, adjust the seed and generate your first set of concepts. If the model was trained successfully, your outputs will appear here.</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="small-muted">FashionForge AI · Built with TensorFlow and Streamlit · Output quality depends on training duration, dataset quality, and GAN convergence.</div>', unsafe_allow_html=True)
