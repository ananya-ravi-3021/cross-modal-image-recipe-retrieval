# Cross-Modal Image-Recipe Retrieval with a Focus on Allergies and Dietary Restrictions

## Link to Paper: 

Multi-label classification of allergens and dietary restrictions directly from a food image, with no recipe text at inference time.

Given a photo of a dish, the model predicts which of 10 allergen/restriction categories are present:

`dairy` · `egg` · `gluten` · `sugar` · `peanut` · `tree nut` · `shellfish` · `finned fish` · `soy` · `meats`

This is the code for my B.Comp. undergraduate dissertation at the National University of Singapore (School of Computing, 2023/24).

The cross-modality is in the supervision, not the inference. Recipe text is used to derive the ground-truth allergen labels for each image; the trained model then works from the image alone. That reflects the practical setting: a diner looking at a plate has no recipe, and allergens are frequently mixed into a dish rather than visible on its surface.

---

## Approach

**Weak supervision from recipes.** Labels are not hand-annotated. `dataset_parsing.py` maps each recipe's ingredient list onto the 10 categories using the keyword dictionary in `utils.py` (`RestrictionsInfo`). For example, any of `{butter, cream, milk, cheese, yogurt, ...}` in the ingredients sets the `dairy` bit.

**Three architectures**, swappable in `train.py`:

| Model | File | Idea |
|---|---|---|
| Naïve multi-label (NML) | `model/naive_model.py` | Pretrained ResNet backbone, 10-way sigmoid head (binary relevance) |
| ResNet-CSRA | `model/resnet_csra.py`, `model/csra.py` | Adds a class-specific spatial attention score to global average pooling |
| ML-GCN | `model/gcn.py` | ResNet-101 plus a 2-layer GCN over a label graph, so co-occurring allergens inform each other |

The ML-GCN label graph is built from two precomputed inputs:
- `model/prob_matrix.py` → a 10×10 label co-occurrence matrix from the training set, binarised by a threshold (`cooccurrence_data.pkl`)
- `text_embedding.py` → a 10×40 feature matrix from BERT embeddings of 40 hand-written food-specific prompts, one slot per allergen label (`feature_data.pkl`)

**Three loss functions**, also swappable: plain `BCELoss`, Asymmetric Loss (`loss/asl_loss.py`), and Class-Adaptive Asymmetric Loss (`loss/caal.py`). The latter two exist because the label matrix is heavily negative-dominated: most dishes carry only 2 or 3 of the 10 categories, and none carry more than 7.

**Class imbalance.** In the original datasets, `peanut`, `shellfish`, `finned fish` and `soy` were badly underrepresented. The fix was to mine Recipe1M+ for recipes containing those labels and append them to the training set, adding 44,703 entries. `clone_minority.py` is the in-repo utility for the same idea, duplicating rows that carry a minority label.

**Semi-supervised extension.** `label_semi_supervised.py` pseudo-labels unlabelled Recipe1M images with a trained model and appends them to the training pool; `train_ssl.py` retrains on the combined set. Self-training ran for 8 iterations, pseudo-labelling one eighth of the unlabelled pool each time.

---

## Repository layout

```
dataset_parsing.py         Build labelled CSVs from recipe datasets
clone_minority.py          Duplicate rare-allergen rows; train/val split
food_dataset.py            PyTorch Dataset + image transforms (224x224)
utils.py                   Allergen->ingredient dictionary; all file paths
train.py                   Supervised training + evaluation loop
train_ssl.py               Training on labelled + pseudo-labelled data
label_semi_supervised.py   Pseudo-labelling of unlabelled images
text_embedding.py          BERT embeddings of allergen prompts -> feature_data.pkl
imagenet_templates.py      The 40 food prompts used above
model/                     naive_model, csra / resnet_csra, gcn, prob_matrix
loss/                      asl_loss (ASL), caal (CAAL)
```

## Data (not included, and this repo is not runnable without it)

**The datasets cannot be redistributed here.** They run to hundreds of gigabytes of images and are covered by their original licences, so they are neither committed to this repo nor downloadable from it. Nothing here will execute end-to-end on a fresh clone: the code is published as a record of the method and implementation, not as a reproducible pipeline.

The labelled set was assembled in stages:

| Folder | Source | Entries |
|---|---|---|
| `food_10k` | Food10k, Kaggle (source no longer available) | 9,653 |
| `food_5k` | [Recipes5k](https://www.kaggle.com/datasets/terry9a9/recipes5k) | 4,826 |
| `food_13k` | [Food Ingredients and Recipe Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images) | 13,582 |
| `food_1M` | [Recipe1M+](https://pic2recipe.csail.mit.edu/), minority-label entries only | 44,703 |

The first three combine into **Food28k** (28,061 entries); adding the Recipe1M+ minority entries gives **Food72k** (72,764), which is the set all reported results use.

For the semi-supervised experiments, 291,056 unlabelled images were drawn from Recipe1M, giving a 1:4 labelled-to-unlabelled ratio.

Access to Recipe1M and Recipe1M+ is requested through the project site linked above. The trained checkpoints and the generated `.pkl` label-graph files are absent too, since both are derived from this data.

If you have obtained the datasets independently, the scripts read their locations from environment variables rather than hardcoded paths:

```bash
export DATASET_ROOT=/path/to/datasets     # raw image + json folders
export DATA_DIR=/path/to/data_files       # generated CSVs
export MODEL_DIR=/path/to/model_states    # saved checkpoints
export PKL_DIR=/path/to/gcn_pkl           # co-occurrence + feature pickles
```

Directory names and file structure would still need to match what `dataset_parsing.py` expects.

## Pipeline

Recorded for reference. These steps assume the data described above is already in place.

```bash
# 1. Build labelled CSVs from the recipe datasets
python dataset_parsing.py

# 2. Duplicate rare-class rows, then re-split train/val
python clone_minority.py

# 3. (ML-GCN only) build the label graph inputs
python model/prob_matrix.py
python text_embedding.py

# 4. Train
python train.py
```

Select the architecture and loss by uncommenting the relevant lines near the top of `train.py`.

**Semi-supervised run:**

```bash
python label_semi_supervised.py   # writes pseudo-labelled rows
python train_ssl.py
```

## Evaluation

Reported each epoch on the validation split, with the best-mAP checkpoint saved:

- **mAP**: mean average precision across the 10 classes (primary metric)
- **Macro F1**: unweighted mean of per-class F1 at threshold 0.5
- **Micro F1**: F1 over pooled predictions, so it favours the common classes

The three metrics consistently ranked mAP > Micro F1 > Macro F1, which matches the expectation that mAP is the most forgiving of class imbalance and Macro F1 the least.

## Results

All figures are validation accuracy (%) on Food72k, batch size 16, 10 epochs, SGD with a OneCycleLR schedule.

| Model + loss | mAP | Macro F1 | Micro F1 |
|---|---|---|---|
| NML + BCE | 60.28 | 40.34 | 48.47 |
| **NML + ASL** | **60.86** | 48.40 | **54.97** |
| NML + CAAL | 60.45 | **48.64** | 54.22 |
| CSRA + BCE | 59.76 | 40.18 | 48.89 |
| CSRA + ASL | 59.88 | 48.12 | 53.65 |
| CSRA + CAAL | 59.81 | 48.17 | 53.81 |
| ML-GCN + BCE | 50.67 | 35.56 | 41.96 |

Best hyperparameters found: ResNet-101 backbone; CSRA with H = 1 and λ = 0.2; ML-GCN with co-occurrence threshold t = 0.4; ASL with γ+ = 2, ps = 0.1, and γ− = 5 (NML) or 6 (CSRA).

**What the experiments showed:**

- The single largest gain came from fixing class balance, not from model complexity. Moving from Food10k to Food72k lifted every metric by roughly 8 points.
- ASL improved the F1 scores substantially over BCE (about +8 macro) while barely moving mAP, since mAP is threshold-free and less sensitive to the positive/negative skew.
- CAAL gave no benefit over ASL here. Its per-class negative focusing parameter is driven by a positive-to-negative gradient ratio that already started near 1 for most classes, so it behaved like ASL with extra machinery. Sweeps over β, μ, α and m changed accuracy by well under a point.
- CSRA matched but did not beat the naïve baseline. Its optimal setting (H = 1, small λ) reduces it to close to plain global average pooling. The likely reason is that ingredients in a cooked dish are mixed rather than localised, so per-region attention has little to latch onto.
- ML-GCN underperformed by around 10 points of mAP. Both graph inputs are suspect: the prompts fed to BERT inserted only the allergen name, never the associated ingredients, and the co-occurrence matrix reflects the quirks of the recipe collection rather than general label relationships.
- Semi-supervised self-training reduced accuracy at every confidence threshold tested (0.5 to 0.8, best around 0.6 to 0.7). With supervised accuracy near 60%, pseudo-label errors compound across iterations.

## Requirements

Python 3.8+, a CUDA GPU, and:

```
torch  torchvision  transformers  scikit-learn
pandas  numpy  opencv-python  imageio
networkx  node2vec
```

Experiments were run on NVIDIA 2080 Ti, 3080 Ti, Titan V and Titan XP GPUs. The `CUDA_VISIBLE_DEVICES` value is hardcoded at the top of several scripts, so change it to match your machine.

## Acknowledgements

Third-party components are adapted with attribution in their file headers:

- **CSRA**: Zhu & Wu, *Residual Attention: A Simple but Effective Method for Multi-Label Recognition*, ICCV 2021 ([code](https://github.com/Kevinz-code/CSRA))
- **ML-GCN**: Chen et al., *Multi-Label Image Recognition with Graph Convolutional Networks*, CVPR 2019 ([code](https://github.com/megvii-research/ML-GCN))
- **CAAL**: Luo et al., *Ingredient Prediction via Context Learning Network With Class-Adaptive Asymmetric Loss*, IEEE TIP 2023
- **ASL**: Ben-Baruch et al., *Asymmetric Loss for Multi-Label Classification*, ICCV 2021 ([code](https://github.com/Alibaba-MIIL/ASL))
- **BERT**: Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2019
- **Text embeddings**: adapted from BorLan (Ma et al., ICCV 2023) ([code](https://github.com/BIT-DA/BorLan))
- **Recipe1M+**: Marin et al., *Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images*, TPAMI 2021
