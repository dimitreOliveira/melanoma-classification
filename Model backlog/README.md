## Model backlog (list of the developed model and metrics)
- **Train** and **validation** are the splits using the train data from the competition.
- The competition metric is **{metric}**.
- **Public LB** is the Public Leaderboard score.
- **Private LB** is the Private Leaderboard score.

---

## Models

| Model | Train | Validation | Public LB | Private LB |
|-------|-------|------------|-----------|------------|
| 1-Melanoma-5Fold EfficientNetB0 | 0.930259 | 0.788910 | 0.827 | ??? |
| 2-Melanoma-5Fold EfficientNetB3 | 0.889511 | 0.809916 | 0.831 | ??? |
| 3-Melanoma-5Fold ResNet18 basic augment | 0.951186 | 0.881608 | 0.884 | ??? |
| 4-Melanoma-3Fold ResNet18 basic augment | 0.924984 | 0.861180 | 0.876 | ??? |
| 5-Melanoma-3Fold ResNet18 backbone LR | 0.680994 | 0.678217 | 0.771 | ??? |
| 6-Melanoma-3Fold ResNet18 backbone LR2 | 0.864836 | 0.835206 | 0.870 | ??? |
| 7-Melanoma-3Fold ResNet18 label_smoothing 05 | 0.981039 | 0.869151 | 0.865 | ??? |
| 8-Melanoma-3Fold ResNet18 label_smoothing 1 | 0.967690 | 0.821159 | 0.865 | ??? |
| 9-Melanoma-3Fold ResNet18 label_smoothing 2 | 0.946336 | 0.838434 | 0.889 | ??? |
| 10-Melanoma-3Fold ResNet18 class_weights_2 | 0.943427 | 0.816079 | 0.864 | ??? |
| 11-Melanoma-3Fold ResNet18 class_weights_5 | 0.906789 | 0.835974 | 0.857 | ??? |
| 12-Melanoma-3Fold ResNet18 class_weights_20 | 0.768815 | 0.720954 | 0.834 | ??? |
| 13-Melanoma-3Fold ResNet18 rotations | 0.980640 | 0.861380 | 0.888 | ??? |
| 14-Melanoma-3Fold ResNet18 45_rotations | 0.966637 | 0.819958 | 0.864 | ??? |
| 15-Melanoma-3Fold ResNet18 crop | 0.916702 | 0.858124 | 0.891 | ??? |
| 16-Melanoma-5Fold ResNet18 crop | 0.950231 | 0.880035 | 0.886 | ??? |
| 17-Melanoma-5Fold ResNet18 crop bs_256 | 0.978396 | 0.852653 | 0.889 | ??? |
| 18-Melanoma-5Fold ResNet50 imagenet | 0.975075 | 0.859745 | 0.891 | ??? |
| 19-Melanoma-5Fold ResNet50 imagenet11k-places365ch | 0.988012 | 0.869718 | 0.880 | ??? |
| 20-Melanoma-5Fold ResNet152 imagenet | 0.967132 | 0.846850 | 0.900 | ??? |
| 21-Melanoma-5Fold ResNet152 imagenet11k | 0.950156 | 0.880889 | 0.914 | ??? |
| 22-Melanoma-5Fold ResNet152 imagenet11k 384x384 | 0.977126 | 0.888166 | 0.911 | ??? |
| 23-Melanoma-3Fold ResNet152 imagenet11k 512x512 | 000 | 000 | 0.892 | ??? |
| 24-Melanoma-5Fold ResNet152 imagenet11k 768x768 | 0.927178 | 0.881075 | 0.869 | ??? |
| 25-Melanoma-5Fold ResNet18 | 0.906409 | 0.844973 | 0.892 | ??? |
| 26-Melanoma-5Fold ResNet18 AUC | 0.896782 | 0.858673 | 0.886 | ??? |
| 27-Melanoma-5Fold ResNet18 RAdam | 0.926222 | 0.866615 | 0.889 | ??? |
| 28-Melanoma-5Fold ResNet18 Exponential decay | 0.962505 | 0.872188 | 0.892 | ??? |
| 29-Melanoma-5Fold ResNet18 Cosine decay | 0.914570 | 0.857385 | 0.890 | ??? |
| 30-Melanoma-5Fold ResNet18 Cosine decay WR SWA | 0.925545 | 0.870840 | 0.892 | ??? |
| 31-Melanoma-5Fold ResNet18 Cosine decay WR | 0.946657 | 0.898607 | 0.889 | ??? |
