# CrisisMMD group share

Private to group members who have downloaded CrisisMMD and accepted the CrisisNLP terms of use (https://crisisnlp.qcri.org/terms-of-use.html). Do not put these files in a public repository. Delete them when the project ends.

Raw tweets and images are not included. Download them from https://crisisnlp.qcri.org/crisismmd and place them as follows, next to `cache/` and `prelim/`:

    data/CrisisMMD_v2.0/data_image/
    data/splits/crisismmd_datasplit_all/
    data/splits/crisismmd_datasplit_agreed_label/

## Files

| File | Contents |
|---|---|
| `cache/image_b0.npz` | `image_id`, `feat` (18,082 x 1280). Frozen EfficientNet-B0, pooled, float16 |
| `cache/text_bert.npz` | `tweet_id`, `feat` (16,058 x 768). Frozen BERT, mean pooled over real tokens, float16 |
| `cache/near_duplicates_humanitarian.csv` | 287 cross-split pHash pairs with distances |
| `prelim/prelim_experiment.ipynb` | Feature extraction and logistic regression comparison |
| `prelim/near-duplicate_check.ipynb` | pHash detection code |

Load the features with `np.load(path, allow_pickle=True)` and match rows by `image_id` and `tweet_id`. The ID arrays are object arrays, so only load copies that come from this folder.

## Limits

- Features are pooled only. Cross-attention needs token level features, which must be extracted separately. ResNet-50 and ViT features are also not included.
- Deduplicated splits do not exist yet. The official agreed label splits were used, and 52 test images were only removed when rescoring.
- The pHash settings are a 64-bit hash (`hash_size=8`) and Hamming distance 6 or less. At distance 4 or less, 47 test images are flagged. Some pairs at distance 6 are similar but different images.
- The CSV covers cross-split pairs in the humanitarian agreed label subset only, so it is incomplete as a list of duplicate groups.
- No placeholder ID list exists yet. The proposal figure of 67 counts two groups (IFTTT "Image not found" 34, "Email this" 33). A third "image not found" group of 8 was found later, so the count is at least 75.
- Whether placeholders are dropped or treated as missing images is still undecided.
