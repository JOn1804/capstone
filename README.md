# Steam Review Toxicity Classifier
### Data Science Capstone Project — Adamson University, 2026

A DistilBERT-based text classifier that classifies Steam game reviews into three categories:
- **Neutral** — mild or factual, no strong complaint or attack
- **Constructive Negative** — valid gameplay criticism (balancing, bugs, performance, etc.)
- **Toxic** — pure harassment or personal attacks with no actionable feedback

Fine-tuned on Helldivers 2 Steam reviews using a two-phase transfer learning approach.

---

## Team
| Member | Role |
|        |      |
| Jonard Doctolero    | AI/ML Lead — model training, dataset sourcing, project management, and documentation |

| Aaron Kyle Camagong | Dataset cleaning, labeling, QA, and user manual documentation |

| Zandrojay Allam     | Dataset cleaning, labeling, QA, and user manual documentation |

| Jerome Devaras      | UI/Dashboard development (Gradio) and user manual documentation |

---
---

## Setup Instructions

### 1. Clone the repository
```
git clone https://github.com/JOn1804/capstone.git
cd capstone
```

### 2. Create and activate virtual environment
```
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Download the model
The trained model files are too large for GitHub (255MB).
Download the `phase2_final` folder from Google Drive:

🔗 https://drive.google.com/drive/folders/1W7fx6vMKb9MTrPRVA9Lww_B8R1dvfn29?usp=sharing

Place the downloaded folder here:
```
capstone/
└── models/
    └── phase2_final/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── tokenizer_config.json
```

### 5. Run the app
```
python app.py
```

Open your browser and go to the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

---

## Project Structure
```
capstone/
├── app.py                  # Main UI (Gradio)
├── requirements.txt        # Python dependencies
├── data/                   # Datasets (excluded from GitHub)
├── models/                 # Trained models (excluded from GitHub)
│   └── phase2_final/       # Final fine-tuned DistilBERT model
└── src/
    ├── train.py            # Phase 1 training (Jigsaw dataset)
    ├── train_phase2.py     # Phase 2 fine-tuning (Helldivers 2 reviews)
    ├── 500.py              # Final Phase 2 training script
    ├── scrape_reviews.py   # Steam review scraper
    ├── scrape_toxic.py     # Targeted toxic review scraper
    └── predict.py          # Standalone inference script
```

---

## Model Performance (Phase 2 Final)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Constructive Negative | 0.76 | 0.77 | 0.77 |
| Neutral               | 0.67 | 0.67 | 0.67 |
| Toxic                 | 0.71 | 0.70 | 0.70 |
| **Overall Accuracy** | | | **0.72** |

---

## Dataset
- **Phase 1:** Jigsaw Toxic Comment Classification Challenge (Kaggle) — ~160,000 Wikipedia comments
- **Phase 2:** Manually labeled Helldivers 2 Steam reviews (1,302 reviews — 500 Constructive, 404 Neutral, 398 Toxic)

---

## Known Limitations
- English-language reviews only
- Optimized for live-service gaming communities — may not generalize to all game genres
- Manual text input only — no live Steam API connection in the UI
