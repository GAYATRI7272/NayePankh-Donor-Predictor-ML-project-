# NayePankh Donor Conversion Predictor (Mini ML Project)

Chota Machine Learning project jo predict karta hai ki website visitor donate karega ya nahi.

## Project Details

| Item | Detail |
|------|--------|
| **NGO** | NayePankh Foundation |
| **Problem** | Kaun sa visitor donate karega? |
| **Language** | Python |
| **Time** | 2-3 din me complete |

## Folder Structure

```
Nayepankh ML Project/
├── requirements.txt      # Python packages
├── generate_data.py      # Data banane ke liye
├── train_model.py        # ML models train + graphs
├── data/
│   └── donor_data.csv    # Generated dataset
└── outputs/              # Graphs (auto generate)
    ├── eda_plots.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

## Setup (Pehli baar)

```bash
cd "Nayepankh ML Project"
pip install -r requirements.txt
```

## Run Project

**Step 1 — Data banao:**
```bash
python generate_data.py
```

**Step 2 — Models train karo:**
```bash
python train_model.py
```

## Models Compared

1. **Logistic Regression** — simple, fast baseline
2. **Random Forest** — usually better accuracy

## Features (Dataset columns)

| Column | Meaning |
|--------|---------|
| `pages_visited` | Kitne pages dekhe |
| `time_on_site_sec` | Site par kitna time (seconds) |
| `mobile_user` | Mobile se aaya? (0/1) |
| `viewed_campaign` | Campaign page dekhi? (0/1) |
| `city_kanpur` | Kanpur se hai? (0/1) |
| `from_instagram` | Instagram se aaya? (0/1) |
| `donated` | Donate kiya? (0/1) — **target** |

## NayePankh ke liye Suggestions

1. Campaign page par clear **Donate** button lagao
2. Mobile donation flow simplify karo
3. Instagram se aane wale visitors ko recurring donation promote karo
4. Contact form me **City** aur **Interest** field add karo
5. Impact numbers homepage par dikhao (meals served, people helped)

## Report ke liye

- `outputs/` folder ki 4 graphs report me attach karo
- [nayepankh.com](https://nayepankh.com/) ke screenshots lo
- Model comparison table `train_model.py` run karne par terminal me dikhega

## Note

Yeh project **synthetic (demo) data** use karta hai. Real production me NayePankh ka Razorpay + website analytics data use hoga.
