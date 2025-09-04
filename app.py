# RECYCLIZER/app.py
"""♻️ Recyclizer — Garbage Classification App
   EfficientNet-B0 model for real-time garbage classification.
#    """

import io
import base64
import time
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from explain_utils import generate_gradcam
from model_utils import discover_classes, load_model, predict_one, DEVICE
from ui_utils import plot_probs, run_live_camera

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="♻️ Recyclizer — Garbage Classifier", layout="wide")
st.title("♻️ Recyclizer — Garbage Classification (EfficientNet-B0)")

# -------------------------------
# Session Defaults
# -------------------------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "results" not in st.session_state:
    st.session_state["results"] = []
if "page" not in st.session_state:
    st.session_state["page"] = 0
if "feedback" not in st.session_state:
    st.session_state["feedback"] = []

# -------------------------------
# Cache Helpers
# -------------------------------
@st.cache_data
def get_classes():
    return discover_classes()

@st.cache_resource
def get_model(num_classes: int):
    return load_model(num_classes)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    class_names = get_classes()
    st.write("**Detected Classes:**")
    for c in class_names:
        st.write(f"- {c}")

    topk = st.slider("Top-K Predictions", 1, len(class_names), min(4, len(class_names)))
    show_bar = st.checkbox("Show Probability Chart", value=True)
    show_gradcam = st.checkbox("🔍 Show Grad-CAM (Top Prediction)", value=True)

    # Grad-CAM target layer selection
    target_layer = st.selectbox(
        "Select Grad-CAM Layer",
        ["features.2", "features.4", "features.6"],
        index=2,
    )

    # Export Section
    if st.session_state["results"]:
        df_side = pd.DataFrame(st.session_state["results"])
        st.subheader("📥 Export Predictions")

        st.markdown(
            """
            <style>
            div.stButton > button, div[data-testid="stDownloadButton"] > button {
                width: 100% !important;
                height: 3em;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            label="⬇️ Download CSV",
            data=df_side.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
        st.download_button(
            label="⬇️ Download JSON",
            data=df_side.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="predictions.json",
            mime="application/json",
        )
        if st.button("🗑️ Clear Results"):
            st.session_state["results"] = []
            st.session_state["uploader_key"] += 1
            st.session_state["page"] = 0
            st.session_state["feedback"] = []
            st.rerun()

        # Quick Summary
        st.subheader("📊 Quick Summary")
        try:
            top_class = df_side["class"].value_counts().idxmax()
            st.write(f"🏆 Most Predicted: **{top_class}**")
        except Exception:
            st.write("No predictions yet.")
        st.metric("Total Predictions", len(df_side))
        st.metric("Unique Classes", df_side["class"].nunique() if not df_side.empty else 0)

# -------------------------------
# Load Model
# -------------------------------
try:
    model = get_model(len(class_names))
    st.success(f"✅ Model loaded on `{DEVICE}`")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -------------------------------
# File Upload Section
# -------------------------------
st.subheader("📤 Upload Image(s)")

files = st.file_uploader(
    "Choose image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}",
)

st.markdown(
    """
    <style>
    [data-testid="stFileUploader"] ul {display: none !important;}
    [data-testid="stFileUploader"] section div div:nth-of-type(2) {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Prediction Section
# -------------------------------
if files:
    all_files = list(files)
    files_per_page = 3
    total_pages = (len(all_files) - 1) // files_per_page + 1
    st.session_state["page"] = min(st.session_state["page"], total_pages - 1)

    start = st.session_state["page"] * files_per_page
    end = start + files_per_page
    page_files = all_files[start:end]

    st.markdown("### 📂 Files on this page")
    for f in page_files:
        st.write(f"- {f.name} ({round(len(f.getbuffer()) / 1024, 1)} KB)")

    st.info(f"Page {st.session_state['page'] + 1} of {total_pages}")

    nav1, _, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅️ Prev", disabled=st.session_state["page"] == 0):
            st.session_state["page"] -= 1
            st.rerun()
    with nav3:
        if st.button("Next ➡️", disabled=st.session_state["page"] >= total_pages - 1):
            st.session_state["page"] += 1
            st.rerun()

    page_results = []

    for f in page_files:
        with st.expander(f"🖼️ {f.name}", expanded=True):
            try:
                f.seek(0)
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                start_time = time.time()
                preds, probs = predict_one(model, img, class_names, topk=topk)
                latency = (time.time() - start_time) * 1000

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                img_data_uri = f"data:image/png;base64,{img_b64}"

                # Keep only Top-1 prediction
                best_class, best_prob = preds[0], probs[0]
                result = {
                    "filename": f.name,
                    "image": img_data_uri,
                    "class": best_class,
                    "probability": float(best_prob),
                    "latency": latency,
                    }

                if not any(r["filename"] == f.name for r in st.session_state["results"]):
                  st.session_state["results"].append(result)


                top_class_idx = class_names.index(preds[0])
                c1, c2, c3 = st.columns([2, 2, 3])
                with c1:
                    st.image(img, caption=f"Original: {f.name}", use_container_width=True)
                with c2:
                    st.markdown("### Predictions")
                    for p, pr in zip(preds, probs):
                        st.write(f"**{p}** ({pr:.1%})")
                        st.progress(float(pr))
                    st.metric("⏱ Inference Time", f"{latency:.2f}ms")
                with c3:
                    if show_bar:
                        plot_probs(preds, probs)

                if show_gradcam:
                    try:
                        gradcam_img = generate_gradcam(model, img, target_layer=target_layer, class_idx=top_class_idx)
                        g1, g2 = st.columns(2)
                        with g1:
                            st.image(img, caption="Uploaded Image", use_container_width=True)
                        with g2:
                            st.image(gradcam_img, caption=f"Grad-CAM → {preds[0]}", use_container_width=True)
                    except Exception as gex:
                        st.error(f"Grad-CAM failed: {gex}")

                # Feedback buttons
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    if st.button(f"✅ Correct ({f.name})"):
                        st.session_state["feedback"].append((f.name, preds[0], True))
                with fb_col2:
                    if st.button(f"❌ Wrong ({f.name})"):
                        st.session_state["feedback"].append((f.name, preds[0], False))

            except Exception as ex:
                st.error(f"Error processing {f.name}: {ex}")

    # Append results instead of overwrite
    st.session_state["results"] += page_results

else:
    st.info("Upload an image to start classification.")

# -------------------------------
# 📊 Summary Dashboard
# -------------------------------
if files and st.session_state["results"]:
    st.divider()
    st.subheader("📊 Summary Dashboard")

    threshold = st.slider("🔎 Confidence Threshold (%)", 0, 100, 50, step=5) / 100.0
    df = pd.DataFrame(st.session_state["results"])
    df["probability"] = df["probability"].astype(float)

    filtered_df = df[df["probability"] >= threshold]
    discarded_df = df[df["probability"] < threshold]

    st.info(
        f"✅ Kept: {len(filtered_df)} predictions | ❌ Discarded: {len(discarded_df)} (below threshold)"
    )

    if filtered_df.empty:
        st.warning("⚠️ No predictions met the confidence threshold.")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "📋 Predictions Table",
                "📊 Class Distribution",
                "🥧 Class Proportion",
                "📈 Avg Confidence",
                "⚠️ Low-Confidence Images",
                "📉 Feedback Stats",
            ]
        )

        with tab1:
            df_local = filtered_df.copy()
            df_local["Thumbnail"] = df_local["image"].apply(
                lambda x: f'<img src="{x}" width="80">'
            )
            df_local = df_local[["Thumbnail", "filename", "class", "probability", "latency"]]
            df_local["probability"] = df_local["probability"].apply(lambda p: f"{p:.1%}")
            st.markdown(df_local.to_html(escape=False, index=False), unsafe_allow_html=True)

        with tab2:
            class_counts = filtered_df["class"].value_counts()
            fig, ax = plt.subplots(figsize=(3, 2))
            class_counts.plot(kind="bar", ax=ax)
            ax.set_title("Predicted Class Distribution", fontsize=10)
            ax.set_ylabel("Count")
            st.pyplot(fig,use_container_width=False)

        with tab3:
            class_counts = filtered_df["class"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(3, 2))
            class_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
            ax2.set_ylabel("")
            ax2.set_title("Class Proportion", fontsize=10)
            st.pyplot(fig2,use_container_width=False)

        with tab4:
            avg_probs = (
                filtered_df.groupby("class")["probability"].mean().sort_values(ascending=False)
            )
            st.bar_chart(avg_probs,use_container_width=True)

        with tab5:
            if discarded_df.empty:
                st.success("🎉 No low-confidence images!")
            else:
                for _, row in discarded_df.iterrows():
                    st.image(row["image"], caption=f"{row['filename']} | {row['probability']:.1%}", width=120)

        with tab6:
            if not st.session_state["feedback"]:
                st.info("No feedback given yet.")
            else:
                fb_df = pd.DataFrame(st.session_state["feedback"], columns=["File", "Prediction", "Correct"])
                st.write(fb_df)
                st.metric("👍 Correct", fb_df["Correct"].sum())
                st.metric("👎 Wrong", (~fb_df["Correct"]).sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                label="📥 Download Filtered CSV",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="filtered_predictions.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                label="📥 Download All CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="all_predictions.csv",
                mime="text/csv",
            )
        with c3:
            st.download_button(
                label="📥 Download All JSON",
                data=df.to_json(orient="records", indent=2).encode("utf-8"),
                file_name="all_predictions.json",
                mime="application/json",
            )

# -------------------------------
# Webcam Section
# -------------------------------
st.divider()
with st.expander("🎥 Try Live Camera Mode", expanded=False):
    try:
        run_live_camera(model, class_names)
    except Exception as cam_err:
        st.error(f"Camera init failed: {cam_err}")
        st.info("Tip: If running on localhost, try with HTTPS or disable XSRF protection.")
