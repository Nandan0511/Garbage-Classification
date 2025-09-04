import time
import numpy as np
import av
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import torch

from model_utils import build_transform, DEVICE

# -------------------------------
# Plotting
# -------------------------------
def plot_probs(class_names, probs):
    fig, ax = plt.subplots()
    ax.barh(class_names, probs, color=plt.cm.viridis(probs))
    ax.set_xlim([0, 1])
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Probabilities")
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center")
    st.pyplot(fig)


# -------------------------------
# Webcam helpers
# -------------------------------
class FPSLimiter:
    def __init__(self, target_fps=5):
        self.min_interval = 1.0 / max(1, target_fps)
        self.last = 0.0
    def ready(self):
        now = time.time()
        if now - self.last >= self.min_interval:
            self.last = now
            return True
        return False


def _load_font(size=20):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


def run_live_camera(model, class_names):
    st.subheader("🎥 Live Camera Mode (Real-time Classification)")
    st.caption("Uses your webcam locally in the browser via WebRTC. No video is uploaded to a server.")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        topk_live = st.slider("Top-K (live)", 1, len(class_names), min(3, len(class_names)), key="topk_live")
    with colB:
        conf_thresh = st.slider("Min confidence", 0.0, 1.0, 0.30, 0.01, key="conf_thresh")
    with colC:
        target_fps = st.slider("Target FPS", 1, 15, 5, key="fps")
    with colD:
        show_overlay = st.checkbox("Show overlay", value=True, key="overlay")

    limiter = FPSLimiter(target_fps=target_fps)
    font = _load_font(22)
    tfm = build_transform()

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        rgb = bgr[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        overlay_lines = []
        if limiter.ready():
            with torch.no_grad():
                x = tfm(pil_img).unsqueeze(0).to(DEVICE)
                logits = model(x)
                pr = torch.softmax(logits, dim=1).cpu().squeeze(0)
                vals, idxs = torch.topk(pr, k=min(topk_live, len(class_names)))
                preds = [class_names[i] for i in idxs.tolist()]
                probs = vals.tolist()

            paired = [(c, p) for c, p in zip(preds, probs) if p >= conf_thresh]
            preds, probs = zip(*paired) if paired else ([], [])
            if show_overlay:
                overlay_lines = [f"{c} — {p:.0%}" for c, p in zip(preds, probs)] or ["Low confidence"]

        if show_overlay:
            draw_img = pil_img.copy()
            draw = ImageDraw.Draw(draw_img)
            box_w, box_h = 420, 30 + 28 * max(1, len(overlay_lines))
            margin = 12
            x0, y0, x1, y1 = margin, margin, margin + box_w, margin + box_h
            draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 160))
            draw.text((x0 + 10, y0 + 6), "Predictions", fill=(255, 255, 255), font=font)

            y = y0 + 34
            for line in overlay_lines:
                draw.text((x0 + 10, y), line, fill=(200, 255, 200), font=font)
                y += 26

            rgb_overlay = np.array(draw_img)
            bgr = rgb_overlay[:, :, ::-1]

        return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    webrtc_streamer(
        key="recyclizer-live",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
