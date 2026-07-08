import gradio as gr
import pandas as pd
import random

# ---------------------------------------------------------------------------
# MODEL INTERFACE
# ---------------------------------------------------------------------------
# This mirrors capstone/src/predict.py from the team GitHub repo exactly, so
# there's zero translation needed once Jon confirms models/phase2 is trained.
#
# CONTRACT (must not change, the rest of the UI depends on this shape):
#   classify(text: str) -> (label: str, confidence: float)
#   - label must be exactly one of: "Toxic", "Constructive Negative", "Neutral"
#   - confidence is a float between 0 and 1
#
# HOW TO SWITCH ON THE REAL MODEL:
#   1. Copy the models/phase2 folder (produced by train_phase2.py) into this
#      project, or point MODEL_DIR at wherever it lives.
#   2. Set USE_DUMMY_MODEL = False
#   3. That's it — load_model() and classify() below match predict.py exactly.
#
# If USE_DUMMY_MODEL is True (default), a keyword-based fake classifier is used
# instead, so the rest of the team can build/test the UI without waiting on
# training to finish. NOTE: as of now it's unconfirmed whether models/phase2
# has actually finished training — check with Jon before flipping this off.

USE_DUMMY_MODEL = False

# Resolves to capstone/models/phase2 no matter where you run `python app.py`
# from (capstone/ or capstone/src/) — avoids the classic "works on my machine,
# FileNotFoundError on yours" bug caused by relative-path assumptions.
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_THIS_DIR, "..", "models", "phase2")

# Matches predict.py's label_names exactly — do not reorder without checking
# with Jon, since this must match however train.py / train_phase2.py encoded
# labels during training (0/1/2 -> Neutral/Constructive Negative/Toxic).
LABEL_MAP = {
    0: "Neutral",
    1: "Constructive Negative",
    2: "Toxic",
}

_model = None
_tokenizer = None
_device = None


def load_model():
    """Loads the fine-tuned model once, on first real use. No-op in dummy mode."""
    global _model, _tokenizer, _device
    if USE_DUMMY_MODEL or _model is not None:
        return

    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
    _model.eval()


def classify(text: str):
    """Single entry point the rest of the app calls. Returns (label, confidence)."""
    if USE_DUMMY_MODEL:
        return _dummy_classify(text)

    load_model()
    import torch

    # padding='max_length', max_length=128 matches predict.py exactly —
    # keeping this identical to how Jon evaluated the model avoids any
    # train/inference mismatch.
    inputs = _tokenizer(
        text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs))
    confidence = float(probs[pred_idx])
    label = LABEL_MAP[pred_idx]
    return label, round(confidence, 2)


# --- dummy fallback, only used while USE_DUMMY_MODEL = True ---
TOXIC_WORDS = ["trash", "clowns", "hate", "garbage", "idiot", "worst", "stupid"]
NEGATIVE_WORDS = ["broken", "bug", "matchmaking", "patch", "issue", "problem", "crash"]


def _dummy_classify(text: str):
    text_lower = text.lower()
    if any(w in text_lower for w in TOXIC_WORDS):
        return "Toxic", round(random.uniform(0.75, 0.97), 2)
    elif any(w in text_lower for w in NEGATIVE_WORDS):
        return "Constructive Negative", round(random.uniform(0.65, 0.9), 2)
    else:
        return "Neutral", round(random.uniform(0.6, 0.95), 2)


# ---------------------------------------------------------------------------
# COLORS (from our dark mode palette)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#1A1A19",
    "surface1": "#232322",
    "surface2": "#2C2C2A",
    "border": "rgba(255,255,255,0.15)",
    "border_strong": "rgba(255,255,255,0.25)",
    "text_primary": "#F5F4F0",
    "text_secondary": "#B4B2A9",
    "text_muted": "#888780",
    "toxic_bg": "#501313",
    "toxic_text": "#F09595",
    "warn_bg": "#412402",
    "warn_text": "#FAC775",
    "neutral_bg": "#2C2C2A",
    "neutral_text": "#B4B2A9",
}

CUSTOM_CSS = f"""
.gradio-container {{
    background: {COLORS['bg']} !important;
}}
#sidebar {{
    background: {COLORS['surface1']};
    border-radius: 12px;
    padding: 1rem;
    border: 0.5px solid {COLORS['border']};
}}
#main-card {{
    background: {COLORS['surface2']};
    border-radius: 12px;
    padding: 1.25rem;
    border: 0.5px solid {COLORS['border']};
}}
.result-card {{
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}}
.result-card p {{
    margin: 0;
}}
.badge {{
    font-size: 11.5px;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 6px;
    display: inline-block;
    white-space: nowrap;
}}
.review-row {{
    border: 0.5px solid {COLORS['border']};
    border-radius: 8px;
    padding: 0.75rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 8px;
    background: {COLORS['surface1']};
}}
.review-text {{
    font-size: 13.5px;
    color: {COLORS['text_primary']};
    flex: 1;
}}
"""


# ---------------------------------------------------------------------------
# HTML RENDER HELPERS
# ---------------------------------------------------------------------------

def render_single_result(label: str, confidence: float):
    if label == "Toxic":
        bg, txt = COLORS["toxic_bg"], COLORS["toxic_text"]
    elif label == "Constructive Negative":
        bg, txt = COLORS["warn_bg"], COLORS["warn_text"]
    else:
        bg, txt = COLORS["neutral_bg"], COLORS["neutral_text"]

    return f"""
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-top:14px;">
      <div class="result-card" style="background:{bg};">
        <p style="font-size:12px; color:{txt}; margin-bottom:4px;">Prediction</p>
        <p style="font-size:20px; font-weight:500; color:{txt};">{label}</p>
      </div>
      <div class="result-card" style="background:{COLORS['surface1']};">
        <p style="font-size:12px; color:{COLORS['text_secondary']}; margin-bottom:4px;">Confidence</p>
        <p style="font-size:20px; font-weight:500; color:{COLORS['text_primary']};">{int(confidence*100)}%</p>
      </div>
    </div>
    """


def badge_colors(label: str):
    if label == "Toxic":
        return COLORS["toxic_bg"], COLORS["toxic_text"]
    elif label == "Constructive Negative":
        return COLORS["warn_bg"], COLORS["warn_text"]
    return COLORS["neutral_bg"], COLORS["neutral_text"]


def render_review_list(df: pd.DataFrame, filter_label: str = "All", limit: int = 20):
    if df is None or df.empty:
        return "<p style='color:#888780; font-size:13px;'>Upload a CSV to see results here.</p>"

    shown = df if filter_label == "All" else df[df["label"] == filter_label]
    rows_html = ""
    for _, row in shown.head(limit).iterrows():
        bg, txt = badge_colors(row["label"])
        rows_html += f"""
        <div class="review-row">
          <p class="review-text">"{row['review']}"</p>
          <span class="badge" style="background:{bg}; color:{txt};">{row['label']} &middot; {int(row['confidence']*100)}%</span>
        </div>
        """
    footer = f"<p style='font-size:12px; color:{COLORS['text_muted']}; margin-top:8px;'>Showing {min(limit, len(shown))} of {len(shown)} {filter_label.lower() if filter_label != 'All' else ''} reviews</p>"
    return rows_html + footer


def render_category_counts(df: pd.DataFrame):
    if df is None or df.empty:
        return "<p style='color:#888780;'>No data yet.</p>", 0, 0, 0, 0
    total = len(df)
    toxic = (df["label"] == "Toxic").sum()
    neg = (df["label"] == "Constructive Negative").sum()
    neutral = (df["label"] == "Neutral").sum()
    return total, toxic, neg, neutral


def render_history(history: list):
    """Renders the sidebar's 'Recent' list from actual classification history,
    newest first. `history` is a list of {"review": str, "label": str} dicts,
    stored in a gr.State so it persists across interactions in the session."""
    if not history:
        return f"<p style='color:{COLORS['text_muted']}; font-size:12px;'>Nothing classified yet.</p>"
    items = ""
    for entry in reversed(history[-15:]):
        _, dot_color = badge_colors(entry["label"])
        snippet = entry["review"][:28] + ("..." if len(entry["review"]) > 28 else "")
        items += f"""
        <div style="display:flex; align-items:center; gap:8px; padding:6px 2px; font-size:12.5px; color:{COLORS['text_secondary']};">
          <span style="width:7px; height:7px; border-radius:50%; background:{dot_color}; flex-shrink:0;"></span>
          <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">"{snippet}"</span>
        </div>
        """
    return items


# ---------------------------------------------------------------------------
# EVENT HANDLERS
# ---------------------------------------------------------------------------

def classify_single(review_text, history):
    if not review_text or not review_text.strip():
        return "<p style='color:#888780; font-size:13px;'>Paste a review first.</p>", history, render_history(history)
    label, conf = classify(review_text)
    history = history + [{"review": review_text.strip(), "label": label}]
    return render_single_result(label, conf), history, render_history(history)


def classify_batch(file, history):
    if file is None:
        return None, "<p style='color:#888780;'>Upload a CSV with a 'review' column.</p>", "0", "0", "0", "0", history, render_history(history)

    df = pd.read_csv(file.name)
    # expects a column literally named "review"; fall back to first column
    col = "review" if "review" in df.columns else df.columns[0]

    labels, confs = [], []
    for text in df[col].astype(str):
        label, conf = classify(text)
        labels.append(label)
        confs.append(conf)
        history = history + [{"review": text.strip(), "label": label}]

    result_df = pd.DataFrame({"review": df[col], "label": labels, "confidence": confs})
    total, toxic, neg, neutral = render_category_counts(result_df)
    return (
        result_df,
        render_review_list(result_df, "All"),
        str(total),
        str(toxic),
        str(neg),
        str(neutral),
        history,
        render_history(history),
    )


def filter_reviews(result_df, label):
    return render_review_list(result_df, label)


# ---------------------------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------------------------

theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill=COLORS["bg"],
    block_background_fill=COLORS["surface2"],
    border_color_primary=COLORS["border"],
)

with gr.Blocks(title="Steam Toxicity Classifier") as demo:
    batch_state = gr.State(pd.DataFrame())
    history_state = gr.State([])  # list of {"review": str, "label": str}, newest last

    with gr.Row():
        # --- Sidebar ---
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.Markdown("🛡️ **Toxicity Classifier**")
            gr.Button("+ New review", size="sm")
            gr.Markdown("Recent", elem_classes="sidebar-label")
            history_display = gr.HTML(render_history([]))
            gr.Markdown("---")
            gr.Markdown(f"<span style='color:{COLORS['text_muted']}; font-size:11px;'>DistilBERT · fine-tuned</span>")

        # --- Main content ---
        with gr.Column(scale=4, elem_id="main-card"):
            with gr.Tabs():
                with gr.Tab("Single review"):
                    review_input = gr.Textbox(
                        label="Paste a Steam review",
                        placeholder="This game is absolute trash, the devs should...",
                        lines=4,
                    )
                    classify_btn = gr.Button("Classify")
                    single_result = gr.HTML()

                    classify_btn.click(
                        fn=classify_single,
                        inputs=[review_input, history_state],
                        outputs=[single_result, history_state, history_display],
                    )

                with gr.Tab("Batch Upload"):
                    with gr.Row():
                        file_input = gr.File(label="Upload CSV of reviews", file_types=[".csv"])
                        upload_btn = gr.Button("Classify batch")

                    with gr.Row():
                        all_count = gr.Button("All\n0", size="sm")
                        toxic_count = gr.Button("Toxic\n0", size="sm")
                        neg_count = gr.Button("Constructive Negative\n0", size="sm")
                        neutral_count = gr.Button("Neutral\n0", size="sm")

                    review_list = gr.HTML()

                    upload_btn.click(
                        fn=classify_batch,
                        inputs=[file_input, history_state],
                        outputs=[batch_state, review_list, all_count, toxic_count, neg_count, neutral_count, history_state, history_display],
                    )

                    all_count.click(fn=lambda df: filter_reviews(df, "All"), inputs=batch_state, outputs=review_list)
                    toxic_count.click(fn=lambda df: filter_reviews(df, "Toxic"), inputs=batch_state, outputs=review_list)
                    neg_count.click(fn=lambda df: filter_reviews(df, "Constructive Negative"), inputs=batch_state, outputs=review_list)
                    neutral_count.click(fn=lambda df: filter_reviews(df, "Neutral"), inputs=batch_state, outputs=review_list)


if __name__ == "__main__":
    demo.launch(theme=theme, css=CUSTOM_CSS)