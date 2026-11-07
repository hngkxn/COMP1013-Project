# COMP1013-Project
Fusing Text and Images to Classify Disaster Tweets (CrisisMMD)

# Fusing Text and Images to Classify Disaster Tweets (CrisisMMD)

Individual submission for COMP1013 Multi-Modal Data Analysis, Adelaide University.
Author: Huang Kaixin, A3228081. Group proposal: 8. The report, slides and video are submitted separately.

## What this project does

Given a tweet with its image, classify it into one of five humanitarian categories (not humanitarian, other relevant
information, rescue/volunteering/donation, infrastructure and utility damage, affected individuals). I use the humanitarian
task of CrisisMMD (agreed-label subset) and ask whether fusing text and image beats the strongest single modality.

Compared on frozen EfficientNet-B0 (image) and BERT (text) features: image-only, text-only, early fusion, late fusion and
gated fusion, each with logistic regression (LR) and, for most, a small MLP.

## Main results (test set, 903 pairs)

Neural models: mean ± standard deviation over 5 seeds. Logistic regression is deterministic.

| Model | Test macro-F1 | Test accuracy |
|---|---|---|
| LR image | 0.701 | 0.776 |
| LR text | 0.679 | 0.764 |
| LR early fusion | 0.791 | 0.857 |
| LR late fusion | 0.793 | 0.853 |
| MLP image | 0.680 ± 0.014 | 0.775 ± 0.010 |
| MLP text | 0.714 ± 0.016 | 0.783 ± 0.006 |
| MLP early fusion | 0.806 ± 0.013 | 0.871 ± 0.005 |
| MLP late fusion | 0.815 ± 0.027 | 0.862 ± 0.018 |
| Gated fusion | 0.770 ± 0.014 | 0.849 ± 0.006 |

- Fusion beats the strongest single-modality model (MLP text, chosen on dev). Four of five fusion models clear the proposal's bar
  (macro-F1 gain of at least 0.05 with a 95% bootstrap interval above 0, seed-0 run); gated fusion does not.
- Early and late fusion are tied within seed noise. The evidence for the gain over the best text model is moderate.
- Fusion also beats both single modalities on all seven held-out events in a leave-one-event-out test.
- Full tables and discussion are in the report. The result files are in `results/`.

## Repository contents

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_group_features_and_pairs/      shared group work (credited below)
│   └── 02_individual_pipeline.ipynb      my work: deduplicated splits, models, evaluation
└── results/
    ├── main_table.csv
    ├── comparisons.csv
    ├── per_class_f1.csv
    ├── per_event_macroF1.csv
    ├── unseen_event_results.csv
    ├── seed_summary.csv
    ├── learning_curves.png
    └── confusion_matrix.png
```

## Data (not included in this repository)

- Dataset: CrisisMMD v2.0 (Alam et al., 2018), https://crisisnlp.qcri.org/crisismmd.html. A copy is also on the official
  QCRI Hugging Face page, https://huggingface.co/datasets/QCRI/CrisisMMD.
- The terms allow humanitarian research use only and forbid redistributing the content. **This repository contains no tweets,
  images, split files or cached features.** Please download the data yourself and delete it when you are done.
- Splits: the agreed-label split for the humanitarian task (Ofli et al., 2020), 6,126 / 998 / 955 pairs, five classes.
- The notebook expects these files under `/content` (Colab) or the folder set in `BASE`:
  - the official split folders (`task_humanitarian_text_img_agreed_lab_{train,dev,test}.tsv`);
  - `cache/image_b0.npz` (EfficientNet-B0 pooled features, 18,082 x 1,280) and `cache/text_bert.npz` (BERT mean-pooled
    features, 16,058 x 768), made by the notebooks in `notebooks/01_group_features_and_pairs/`;
  - `cache/near_duplicates_humanitarian.csv`, the cross-split near-duplicate pairs (perceptual hash, Hamming distance 6 or
    less), made by the same group notebooks.

## Environment

Python 3.13.18. Packages and versions are in `requirements.txt`. The reported run used a CPU-only Google Colab runtime
(torch 2.11.0, CPU build); no GPU is needed once the features exist.

The cached features were produced in a separate environment by the group notebooks (see Credits), so installing the packages
above is enough to run notebook 02, but not to rebuild the features.

## How to reproduce

1. Download the data and splits (see above) and obtain `cache/` by running the notebooks in
   `notebooks/01_group_features_and_pairs/` on the data.
2. Open `notebooks/02_individual_pipeline.ipynb` (for example in Google Colab) and put the files from step 1 under `/content`.
   Set the paths in the settings cell if you use another folder.
3. **Development run** (train and dev only, test untouched): keep `FINAL_RUN = False`, run all cells. It takes a few minutes on
   CPU. Expected checks:
   - original split sizes 6,126 / 998 / 955;
   - "pairs used: 287 of 287";
   - after deduplication 6,235 / 941 / 903 (52 test and 62 dev rows moved);
   - dev macro-F1: LR early fusion 0.767, MLP early fusion about 0.785.
4. **Final run** (test evaluated once): set `FINAL_RUN = True` and `RUN_SEEDS = True`, restart and run all cells. This writes the
   tables and figures to `results/`. Logistic regression results should match the table above exactly; neural results depend on
   the seeds (0 to 4) and library versions and may differ slightly.

Settings used for the reported results:

| Setting | Value |
|---|---|
| pHash distance threshold | 6 (Hamming distance on a 64-bit hash) |
| Deduplication rule | each near-duplicate group and each tweet moved to one split, train over dev over test |
| Final split sizes | 6,235 / 941 / 903 (train / dev / test) |
| Logistic regression | class-weighted, C in {1e-4, 1e-3, 1e-2, 1e-1, 1} chosen on dev |
| MLP (hidden, dropout, weight decay) | image (64, 0.7, 1e-3), text (256, 0.5, 1e-3), early fusion (64, 0.5, 1e-3) |
| Gated fusion | hidden 128, dropout 0.5, weight decay 1e-3 (not tuned) |
| Selection | all tuning on dev only; test used once in the final run |
| Seeds | 0 to 4 for neural models |

## Notebook outline

Settings, data loading, deduplicated splits, features, models (LR, MLP, gated), evaluation (metrics, bootstrap intervals,
McNemar), per-class and per-event results, confusion matrix, unseen-event (leave-one-event-out) test, learning curves, and a
five-seed summary for the neural models.

## Limitations

Only agreed-label pairs are used, so scores probably overestimate performance on real tweets. The rarest class (affected
individuals) has 9 dev and 9 test samples. Features are frozen and pooled, from one dataset. Bootstrap intervals come from one
seed. Cross-attention was not implemented (it needs token-level features). At least 75 placeholder images were not flagged.

## Credits

- Cached features and near-duplicate pair detection: Wenghua Liu, shared group work used with permission.
- Dataset: Alam, F., Ofli, F. and Imran, M. (2018) CrisisMMD: Multimodal Twitter datasets from natural disasters, ICWSM.
  Agreed-label split: Ofli, F., Alam, F. and Imran, M. (2020), ISCRAM.
- Perceptual hashing: Zauner, C. (2010), Implementation and benchmarking of perceptual image hash functions.

## Ethics and data handling

Tweets can contain personal information. Usernames are masked in preprocessing, no raw tweets or images are shown or
published, and all downloaded data is deleted after marking, as the dataset terms require.
