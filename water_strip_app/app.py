import streamlit as st
import numpy as np
import cv2
from PIL import Image

from src.model_loader import load_models, predict_concentration, UNITS_OF_CONCENTRATION
from src.image_refined import crop_strip_simple, get_pad_data_refined

st.set_page_config(page_title="Water Strip ML Demo", layout="wide")

@st.cache_resource
def cached_models():
    return load_models("models")

models = cached_models()
param_names_sorted = sorted(models.keys())
param_id_to_name = {i+1: n for i, n in enumerate(param_names_sorted)}

st.title("🧪 Water Test Strip Analyzer (Streamlit Demo)")
st.write("Upload strip image → detect pads → map pad order → predict concentrations.")

with st.expander("📌 Available Parameters (IDs)", expanded=False):
    for i, name in enumerate(param_names_sorted, start=1):
        st.write(f"**{i}**: {name}")

# Sidebar controls
num_pads = st.sidebar.number_input("Expected pads on strip", min_value=1, max_value=25, value=16, step=1)

default_seq = ",".join([str(i) for i in range(1, min(num_pads, len(param_names_sorted)) + 1)])
param_sequence = st.sidebar.text_input(
    "Parameter sequence (IDs) - comma separated\nPad1→ID1, Pad2→ID2, ...",
    value=default_seq
)

process_btn = st.sidebar.button("✅ Detect Pads")
predict_btn = st.sidebar.button("🎯 Predict")

uploaded = st.file_uploader("Upload strip image", type=["jpg", "jpeg", "png"])

# session state
if "labs_std" not in st.session_state:
    st.session_state.labs_std = []
if "vis_bgr" not in st.session_state:
    st.session_state.vis_bgr = None
if "strip_bgr" not in st.session_state:
    st.session_state.strip_bgr = None

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

pil_img = Image.open(uploaded)

# Resize image to a reasonable size for processing to avoid memory issues
max_size = (2048, 2048)  # Max width/height
pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)

# Further resize for display
display_img = pil_img.copy()
max_display_size = (800, 600)
display_img.thumbnail(max_display_size, Image.Resampling.LANCZOS)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    st.image(display_img, width='stretch')

if process_btn:
    try:
        raw_bgr = pil_to_bgr(pil_img)
        strip = crop_strip_simple(raw_bgr)
        labs_std, vis = get_pad_data_refined(strip, int(num_pads))

        st.session_state.strip_bgr = strip
        st.session_state.labs_std = labs_std
        st.session_state.vis_bgr = vis

        st.success(f"Detected {len(labs_std)} pads.")
    except Exception as e:
        st.error(f"Detection failed: {e}")

if st.session_state.vis_bgr is not None:
    with col2:
        st.subheader("Detected Pads (Visualization)")
        st.image(bgr_to_pil(st.session_state.vis_bgr), width='stretch')

if predict_btn:
    if not st.session_state.labs_std:
        st.warning("Please detect pads first.")
        st.stop()

    # Parse IDs
    try:
        ids = [int(x.strip()) for x in param_sequence.split(",") if x.strip()]
    except:
        st.error("Invalid sequence. Example: 1,2,3,4")
        st.stop()

    labs_std = st.session_state.labs_std
    n = len(labs_std)

    if len(ids) > n:
        st.warning(f"IDs given: {len(ids)} but pads detected: {n}. Using first {n} IDs.")
        ids = ids[:n]

    if len(ids) < n:
        st.warning(f"Only {len(ids)} IDs provided for {n} pads. Predicting only first {len(ids)} pads.")

    # Validate IDs and predict
    results = []
    for i, pid in enumerate(ids):
        if pid < 1 or pid > len(param_names_sorted):
            st.error(f"Invalid parameter ID {pid}. Allowed range: 1..{len(param_names_sorted)}")
            st.stop()

        pname = param_id_to_name[pid]
        model = models[pname]

        pred = predict_concentration(model, labs_std[i])

        results.append({
            "Pad": i+1,
            "Param ID": pid,
            "Parameter": pname,
            "L*": round(labs_std[i][0], 2),
            "a*": round(labs_std[i][1], 2),
            "b*": round(labs_std[i][2], 2),
            "Prediction": round(pred, 6),
            "Units": UNITS_OF_CONCENTRATION,
        })

    st.subheader("✅ Predictions")
    st.dataframe(results, width='stretch')