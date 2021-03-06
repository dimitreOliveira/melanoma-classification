## Model backlog (list of the developed model and metrics)
- **Train** and **validation** are the splits using the train data from the competition.
- The competition metric is **AUC**.
- **Public LB** is the Public Leaderboard score.
- **Private LB** is the Private Leaderboard score.

---

## Models

| Model | Train | Validation | Public LB | Private LB |
|-------|-------|------------|-----------|------------|
| 1-Melanoma-5Fold EfficientNetB0 | 0.930259 | 0.788910 | 0.827 | 0.8136 |
| 2-Melanoma-5Fold EfficientNetB3 | 0.889511 | 0.809916 | 0.831 | 0.8211 |
| 3-Melanoma-5Fold ResNet18 basic augment | 0.951186 | 0.881608 | 0.884 | 0.8759 |
| 4-Melanoma-3Fold ResNet18 basic augment | 0.924984 | 0.861180 | 0.876 | 0.8769 |
| 5-Melanoma-3Fold ResNet18 backbone LR | 0.680994 | 0.678217 | 0.771 | 0.7506 |
| 6-Melanoma-3Fold ResNet18 backbone LR2 | 0.864836 | 0.835206 | 0.870 | 0.8421 |
| 7-Melanoma-3Fold ResNet18 label_smoothing 05 | 0.981039 | 0.869151 | 0.865 | 0.8467 |
| 8-Melanoma-3Fold ResNet18 label_smoothing 1 | 0.967690 | 0.821159 | 0.865 | 0.8349 |
| 9-Melanoma-3Fold ResNet18 label_smoothing 2 | 0.946336 | 0.838434 | 0.889 | 0.8514 |
| 10-Melanoma-3Fold ResNet18 class_weights_2 | 0.943427 | 0.816079 | 0.864 | 0.8690 |
| 11-Melanoma-3Fold ResNet18 class_weights_5 | 0.906789 | 0.835974 | 0.857 | 0.8416 |
| 12-Melanoma-3Fold ResNet18 class_weights_20 | 0.768815 | 0.720954 | 0.834 | 0.8068 |
| 13-Melanoma-3Fold ResNet18 rotations | 0.980640 | 0.861380 | 0.888 | 0.8796 |
| 14-Melanoma-3Fold ResNet18 45_rotations | 0.966637 | 0.819958 | 0.864 | 0.8456 |
| 15-Melanoma-3Fold ResNet18 crop | 0.916702 | 0.858124 | 0.891 | 0.8798 |
| 16-Melanoma-5Fold ResNet18 crop | 0.950231 | 0.880035 | 0.886 | 0.8750 |
| 17-Melanoma-5Fold ResNet18 crop bs_256 | 0.978396 | 0.852653 | 0.889 | 0.8589 |
| 18-Melanoma-5Fold ResNet50 imagenet | 0.975075 | 0.859745 | 0.891 | 0.8835 |
| 19-Melanoma-5Fold ResNet50 imagenet11k-places365ch | 0.988012 | 0.869718 | 0.880 | 0.8466 |
| 20-Melanoma-5Fold ResNet152 imagenet | 0.967132 | 0.846850 | 0.900 | 0.8706 |
| 21-Melanoma-5Fold ResNet152 imagenet11k | 0.950156 | 0.880889 | 0.914 | 0.8861 |
| 22-Melanoma-5Fold ResNet152 imagenet11k 384x384 | 0.977126 | 0.888166 | 0.911 | 0.8760 |
| 23-Melanoma-3Fold ResNet152 imagenet11k 512x512 | ??? | ??? | 0.892 | 0.8737 |
| 24-Melanoma-1Fold ResNet152 imagenet11k 768x768 | 0.927178 | 0.881075 | 0.869 | 0.8537 |
| 25-Melanoma-5Fold ResNet18 | 0.906409 | 0.844973 | 0.892 | 0.8643 |
| 26-Melanoma-5Fold ResNet18 AUC | 0.896782 | 0.858673 | 0.886 | 0.8724 |
| 27-Melanoma-5Fold ResNet18 RAdam | 0.926222 | 0.866615 | 0.889 | 0.8756 |
| 28-Melanoma-5Fold ResNet18 Exponential decay | 0.962505 | 0.872188 | 0.892 | 0.8853 |
| 29-Melanoma-5Fold ResNet18 Cosine decay | 0.914570 | 0.857385 | 0.890 | 0.8768 |
| 30-Melanoma-5Fold ResNet18 Cosine decay WR SWA | 0.925545 | 0.870840 | 0.892 | 0.8670 |
| 31-Melanoma-5Fold ResNet18 Cosine decay WR | 0.946657 | 0.898607 | 0.889 | 0.8825 |
| 32-Melanoma-5Fold ResNet18 Cosine decay WR2 | 0.931988 | 0.854173 | 0.877 | 0.8674 |
| 33-melanoma-5fold-resnet152 imagenet11k | 0.974451 | 0.877713 | 0.899 | ??? |
| 34-melanoma-5fold-resnet152 imagenet11k class_weights | 0.724695 | 0.722104 | 0.831 | ??? |
| 35-Melanoma-5Fold ResNet18 step decay 05 | 0.992601 | 0.831931 | 0.866 | 0.8256 |
| 36-melanoma-5fold-resnet152 imagenet11k external data | 0.935322 | 0.940607 | 0.916 | ??? |
| 37-melanoma-5fold-resnet152 external data bs_512 | 0.948843 | 0.947044 | 0.860 | ??? |
| 38-melanoma-3fold-resnet152 external data bs_512 | 0.996839 | 0.765615 | 0.818 | ??? |
| 39-melanoma-3fold-EfficientNetB3 70k TTA | 0.997282 | 0.839818 | 0.885 | ??? |
| 40-melanoma-3fold-EfficientNetB3 16-sample dropout | 0.985339 | 0.896662 | 0.901 | ??? |
| 41-melanoma-3fold-EfficientNetB0 16-sample dropout | 0.978523 | 0.893157 | 0.928 | ??? |
| 42-melanoma-3fold-EfficientNetB0 cosine decay wr swa 16-sample | 0.982849 | 0.879591 | 0.929 | ??? |
| 43-melanoma-3fold-EfficientNetB0 2020 cosine wr swa 16-sample | 0.975377 | 0.883311 | 0.907 | ??? |
| 44-melanoma-3fold-EfficientNetB6 16-samples dropout | 0.973873 | 0.903858 | 0.913 | ??? |
| 45-melanoma-3fold-EfficientNetB0 16-samples dropout | 0.979255 | 0.879807 | 0.908 | ??? |
| 46-melanoma-3fold-EfficientNetB0 spatial dropout | 0.983010 | 0.859038 | 0.894 | ??? |
| 47-melanoma-3fold-EfficientNetB0 advanced augmentation | 0.981037 | 0.873423 | 0.926 | ??? |
| 48-melanoma-3fold-EfficientNetB0 resize 224x224 | 0.981341 | 0.873970 | 0.904 | ??? |
| 49-melanoma-3fold-EfficientNetB6 public | 0.960757 | 0.881337 | ??? | ??? |
| 50-melanoma-3fold-EfficientNetB6 custom aug | 0.970015 | 0.893900 | ??? | ??? |
| 51-melanoma-3fold-EfficientNetB6 external data | 0.981421 | 0.893983 | ??? | ??? |
| 52-melanoma-3fold-EfficientNetB6 bs_512 | 0.983456 | 0.873968 | ??? | ??? |
| 53-melanoma-5fold-EfficientNetB0 noisy-student | 0.972530 | 0.870910 | 0.921 | ??? |
| 54-melanoma-5fold-EfficientNetB0 bs_64 | 0.965886 | 0.858830 | ??? | ??? |
| 55-melanoma-5fold-EfficientNetB0 bias initialization | 0.965621 | 0.875473 | 0.921 | ??? |
| 56-melanoma-5fold-EfficientNetB0 bias init bs_512 | 0.986044 | 0.861017 | ??? | ??? |
| 57-melanoma-5fold-EfficientNetB0 class_weights AUC monitor | 0.982143 | 0.907337 | 0.917 | ??? |
| 58-melanoma-5fold-EfficientNetB0 no smoothing | 0.985234 | 0.910786 | 0.917 | ??? |
| 59-melanoma-5fold-EfficientNetB6 | 0.961469 | 0.905375 | 0.910 | 0.8938 |
| 60-melanoma-5fold-EfficientNetB7 | 0.954760 | 0.899402 | ??? | ??? |
| 61-melanoma-5fold-EfficientNetB0 lbl smoothing 02 | 0.981862 | 0.905961 | ??? | ??? |
| 62-melanoma-5fold-InceptionResNetV2 | 0.967108 | 0.894248 | ??? | ??? |
| 63-melanoma-5fold-ResNet18 RAdam | 0.974887 | 0.898184 | ??? | ??? |
| 64-melanoma-5fold-SEResNet18 RAdam | 0.969020 | 0.894261 | 0.897 | 0.8901 |
| 65-melanoma-5fold-EfficientNetB0 RAdam | 0.979056 | 0.904889 | 0.921 | ??? |
| 66-melanoma-5fold-EfficientNetB0 augmentation refator | 0.969065 | 0.906764 | 0.920 | ??? |
| 67-melanoma-5fold-EfficientNetB0 cutout | 0.963932 | 0.901653 | 0.919 | ??? |
| 68-melanoma-5fold-SEResNet18 RAdam cutout | 0.949809 | 0.893800 | ??? | ??? |
| 69-melanoma-5fold-EfficientNetB6 256x256 | 0.961562 | 0.888757 | ??? | ??? |
| 70-melanoma-5fold-EfficientNetB6 384x384 | 0.922855 | 0.890681 | ??? | ??? |
| 71-melanoma-5fold-EfficientNetB6 512x512 | 0.984935 | 0.889693 | ??? | ??? |
| 72-melanoma-1fold-EfficientNetB6 256x256 | 0.941793 | 0.882181 | ??? | ??? |
| 73-melanoma-1fold-EfficientNetB6 step decay | 0.918486 | 0.875922 | 0.901 | ??? |
| 74-melanoma-1fold-EfficientNetB6 imagenet | 0.985203 | 0.866455 | ??? | ??? |
| 75-melanoma-1fold-EfficientNetB6 384x384 public | 0.920580 | 0.888131 | 0.907 | ??? |
| 76-melanoma-1fold-EfficientNetB6 384x384 light augmentation | 0.956563 | 0.902519 | 0.895 | ??? |
| 77-melanoma-1fold-EfficientNetB6 384x384 medium augmentation | 0.944009 | 0.887833 | ??? | ??? |
| 78-melanoma-1fold-EfficientNetB6 384x384 heavy augmentation | 0.921409 | 0.879210 | ??? | ??? |
| 79-melanoma-5fold-EfficientNetB6 384x384 public | 0.914253 | 0.868182 | 0.9219 | ??? |
| 80-melanoma-5fold-EfficientNetB6 384x384 light augmentation | 0.950163 | 0.895704 | ??? | ??? |
| 81-melanoma-5fold-EfficientNetB6 384x384 medium augmentation | 0.941243 | 0.892399 | 0.9131 | ??? |
| 82-melanoma-1fold-EfficientNetB6 384x384 improved augmentation | 0.934016 | 0.905199 | 0???0 | ??? |
| 83-melanoma-1fold-EfficientNetB4 384x384 improved aug | 0.943513 | 0.908020 | ??? | ??? |
| 84-melanoma-5fold-EfficientNetB6 fixed | ??? | 0.903 | 0.9359 | ??? |
| 85-melanoma-5fold-EfficientNetB6 fixed | ??? | 0.909 | 0.9324 | ??? |
| 86-melanoma-5fold-EfficientNetB4 384 | ??? | 0.902 | 0.9282 | ??? |
| 87-melanoma-5fold-EfficientNetB6 RAdam 2xTTA | ??? | 0.896 | 0.9365 | 0.9156 |
| 88-melanoma-5fold-EfficientNetB6 Adam 2xTTA | ??? | 0.902 | 0.9322 | 0.9180 |
| 89-melanoma-5fold-EfficientNetB6 step decay basic aug | ??? | 0.908 | 0.9324 | ??? |
| 90-melanoma-5fold-EfficientNetB6 256x256 | ??? | 0.892 | 0.9291 | 0.9095 |
| 91-melanoma-5fold-EfficientNetB6 256x256 bias | ??? | 0.898 | 0.9229 | 0.9148 |
| 92-melanoma-5fold-EfficientNetB6 256x256 | ??? | 0.887 | 0.9342 | 0.9156 |
| 93-melanoma-5fold-EfficientNetB6 256x256 | ??? | 0.886 | 0.9403 | 0.9175 |
| 94-melanoma-5fold-EfficientNetB6 256 improved aug | ??? | 0.878 | 0.9281 | 0.9137 |
| 95-melanoma-5fold-EfficientNetB6 256 improved aug2 | ??? | 0.896 | 0.9275 | 0.9225 |
| 96-melanoma-5fold-EfficientNetB6 256 improved aug3 | ??? | 0.912 | 0.9312 | 0.9135 |
| 97-melanoma-5fold-EfficientNetB6 256 improved aug4 | ??? | 0.904 | 0.9310 | 0.9124 |
| 98-melanoma-5fold-EfficientNetB4 256 | ??? | 0.897 | 0.9334 | 0.9155 |
| 99-melanoma-5fold-EfficientNetB5 256 | ??? | 0.905 | 0.9217 | ??? |
| 100-melanoma-5fold-EfficientNetB6 256 | ??? | 0.906 | 0.9305 | 0.9215 |
| 101-melanoma-5fold-EfficientNetB6 256 imp aug5 | ??? | 0.896 | 0.9330 | 0.9174 |
| 102-melanoma-5fold-EfficientNetB4 384 | ??? | 0.888 | 0.9303 | 0.9063 |
| 103-melanoma-5fold-EfficientNetB4 384 | ??? | 0.898 | 0.9258 | ??? |
| 104-melanoma-5fold-EfficientNetB4 384 random | ??? | 0.891 | 0.9307 | ??? |
| 105-melanoma-5fold-EfficientNetB6 256 class weights | ??? | 0.914 | 0.9296 | ??? |
| 106-melanoma-5fold-EfficientNetB6 256 class weights v2 | ??? | 0.913 | 0.9321 | ??? |
| 107-melanoma-5fold-EfficientNetB6 256 external data | ??? | 0.900 | 0.9281 | ??? |
| 108-melanoma-5fold-EfficientNetB6 256 2018 data | ??? | 0.905 | 0.9280 | ??? |
| 109-melanoma-5fold-EfficientNetB6 256 2019 data | ??? | 0.898 | ??? | ??? |
| 110-melanoma-5fold-EfficientNetB6 256 cutout aug | ??? | 0.912 | 0.9403 | 0.9217 |
| 111-melanoma-5fold-EfficientNetB6 256 2018 fold | ??? | 0.915 | 0.9206 | ??? |
| 112-melanoma-5fold-EfficientNetB6 256 2019 fold | ??? | 0.898 | ??? | ??? |
| 113-melanoma-5fold-EfficientNetB6 256 light TTA | ??? | 0.905 | 0.9322 | 0.9204 |
| 114-melanoma-5fold-EfficientNetB6 256 external fold | ??? | 0.900 | 0.9353 | ??? |
| 115-melanoma-5fold-EfficientNetB6 256 malig_2020 | ??? | 0.892 | ??? | ??? |
| 116-melanoma-5fold-EfficientNetB6 256 malig_2020ne | ??? | 0.907 | ??? | ??? |
| 117-melanoma-5fold-EfficientNetB6 256 malig_all | ??? | 0.916 | 0.9227 | 0.9295 |
| 118-melanoma-5fold-EfficientNetB6 256 2018 malig | ??? | 0.909 | 0.9374 | ??? |
| 119-melanoma-5fold-EfficientNetB4 oversamp cosine WR | ??? | 0.926 | 0.9485 | ??? |
| 120-melanoma-5fold-EfficientNetB4 256 2018 weighted | ??? | 0.909 | 0.9416 | ??? |
| 121-melanoma-5fold-EfficientNetB4 256 weighted | ??? | 0.918 | 0.9324 | ??? |
| 122-melanoma-5fold-EfficientNetB4 384 weighted | ??? | 0.911 | 0.9246 | 0.9158 |
| 123-melanoma-5fold-EfficientNetB4 256 2020x2 | ??? | 0.910 | 0.9297 | 0.9194 |
| 124-melanoma-5fold-EfficientNetB4 256 2020x3 | ??? | 0.905 | 0.9329 | 0.9199 |
| 125-melanoma-5fold-EfficientNetB4 256 oversampling | ??? | 0.910 | 0.9469 | 0.9335 |
| 126-melanoma-5fold-EfficientNetB4 256 oversamp+aug | ??? | 0.921 | 0.9432 | 0.9344 |
| 127-melanoma-5fold-EfficientNetB4 3x_ds oversamp | ??? | 0.918 | 0.9419 | 0.9331 |
| 128-melanoma-5fold-EfficientNetB4 oversamp cosineW | ??? | 0.925 | 0.9409 | 0.9348 |
| 129-melanoma-5fold-EfficientNetB4 bs_16 384 | ??? | 0.934 | 0.9495 | 0.9362 |
| 130-melanoma-5fold-EfficientNetB4 bs_16 | ??? | 0.927 | 0.9391 | ??? |
| 131-melanoma-5fold-EfficientNetB4 bs_16 384 | ??? | 0.930 | 0.9508 | 0.9350 |
| 132-melanoma-5fold-EfficientNetB4 512 | ??? | 0.936 | 0.9503 | ??? |
| 133-melanoma-5fold-EfficientNetB4 384 oversamp_05 | ??? | 0.918 | ??? | ??? |
| 134-melanoma-5fold-EfficientNetB4 bs_16 512 | ??? | 0.934 | 0.9569 | 0.9348 |
| 135-melanoma-5fold-EfficientNetB0 512 | ??? | 0.921 | ??? | ??? |
| 136-melanoma-5fold-EfficientNetB5 512 | ??? | 0.937 | 0.9470 | 0.9396 |
| 137-melanoma-5fold-EfficientNetB4 512 | ??? | 0.935 | 0.9473 | ??? |
| 138-melanoma-5fold-EfficientNetB4 ds sampling 512 | ??? | 0.922 | 0.9515 | ??? |
| 139-melanoma-5fold-EfficientNetB4 512 | ??? | 0.940 | 0.9455 | ??? |
| 140-melanoma-5fold-EfficientNetB4 one_cycle cosine 512 | ??? | 0.937 | 0.9477 | ??? |
| 141-melanoma-5fold-EfficientNetB3 one_cycle cosine 256 | ??? | 0.913 | ??? | ??? |
| 142-melanoma-5fold-EfficientNetB3 one_cycle dropout 256 | ??? | 0.926 | 0.9397 | ??? |
| 143-melanoma-5fold-EfficientNetB3 focal loss 256 | ??? | 0.925 | 0.9371 | ??? |
| 144-melanoma-5fold-EfficientNetB3 focal reg samp 256 | ??? | 0.922 | ??? | ??? |
| 145-melanoma-5fold-EfficientNetB3 one_cycle dropout 384 | ??? | 0.931 | 0.9437 | ??? |
| 146-melanoma-5fold-EfficientNetB3 one_cycle 512 | ??? | 0.931 | 0.9483 | 0.9251 |
| 147-melanoma-5fold-EfficientNetB1 one_cycle 512 | ??? | 0.930 | 0.9512 | 0.9255 |
| 148-melanoma-5fold-EfficientNetB3 one_cycle 384 | ??? | 0.927 | ??? | ??? |
| 149-melanoma-5fold-EfficientNetB3 meta fts 384 | ??? | 0.928 | 0.9498 | 0.9367 |
| 150-melanoma-5fold-EfficientNetB3 one_cycle dropout 512 | ??? | 0.933 | 0.9448 | ??? |
| 151-melanoma-5fold-EfficientNetB3 meta fts 384 | ??? | 0.930 | 0.9507 | 0.9324 |
| 152-melanoma-5fold-EfficientNetB4 cosine_WR 512 | ??? | 0.933 | 0.9471 | ??? |
| 153-melanoma-5fold-EfficientNetB3 meta fts 512 | ??? | 0.929 | 0.9507 | 0.9321 |
| 154-melanoma-5fold-EfficientNetB0 224 | ??? | 0.884 | ??? | ??? |
| 155-melanoma-5fold-EfficientNetB0 meta fts 224 | ??? | 0.890 | ??? | ??? |
| 156-melanoma-5fold-EfficientNetB4 2019_data 512 | ??? | 0.931 | 0.9486 | ??? |
| 157-melanoma-5fold-EfficientNetB4 ds_sampling 512 | ??? | 0.938 | 0.9486 | ??? |
| 159-melanoma-5fold-EfficientNetB4 2019 380 | ??? | 0.931 | 0.9499 | ??? |
| 160-melanoma-5fold-EfficientNetB4 380 | ??? | 0.932 | ??? | ??? |
| 161-melanoma-5fold-EfficientNetB5 MixUp no_cut 456 | ??? | 0.935 | ??? | ??? |
| 162-melanoma-5fold-EfficientNetB0 SGD bs_128 224 | ??? | 0.874 | ??? | ??? |
| 163-melanoma-5fold-EfficientNetB4 380 | ??? | 0.931 | ??? | ??? |
| 164-melanoma-5fold-EfficientNetB4 MixUp 380 | ??? | 0.930 | ??? | ??? |
| 165-melanoma-5fold-EfficientNetB4 MixUp no_lbl 380 | ??? | 0.925 | ??? | ??? |
| 166-melanoma-5fold-EfficientNetB4 MixUp range 380 | ??? | 0.920 | ??? | ??? |
| 167-melanoma-5fold-EfficientNetB3 300 | ??? | 0.926 | ??? | ??? |
| 168-melanoma-5fold-EfficientNetB4 456 | ??? | 0.929 | ??? | ??? |
| 169-melanoma-5fold-EfficientNetB4 MixUp no_cut 380 | ??? | 0.935 | ??? | ??? |
| 171-melanoma-5fold-EfficientNetB0 Warm cutout 224 | ??? | 0.891 | ??? | ??? |
| 172-melanoma-5fold-EfficientNetB0 MixUp_1 224 | ??? | ??? | ??? | ??? |
| 173-melanoma-5fold-EfficientNetB0 MixUp_04 224 | ??? | ??? | ??? | ??? |



### XGBM

| Model | Train | Validation | Public LB | Private LB |
|-------|-------|------------|-----------|------------|
| 1-melanoma-5fold-XGBM baseline | 0.646091 | 0.656711 | ??? | ??? |
| 2-melanoma-5fold-XGBM mean | 0.642615 | 0.652225 | ??? | ??? |
| 3-melanoma-5fold-XGBM label encoding | 0.661438 | 0.669618 | ??? | ??? |
| 4-melanoma-5fold-XGBM image features | 0.778892 | 0.800004 | ??? | ??? |
| 5-melanoma-5fold-XGBM image features v2 | 0.745335 | 0.762053 | ??? | ??? |
| 6-melanoma-5fold-XGBM combined fts | 0.784281 | 0.798628 | ??? | ??? |
| 7-melanoma-5fold-XGBM no_anotom | 0.782082 | 0.798661 | ??? | ??? |
| 8-melanoma-5fold-XGBM no_mean color | 0.746723 | 0.754998 | ??? | ??? |
| 9-melanoma-5fold-XGBM random param search | 0.780412 | 0.847426 | ??? | ??? |
| 10-melanoma-5fold-XGBM basic features | 0.679200 | 0.680989 | ??? | ??? |
| 11-melanoma-5fold-XGBM basic fts external data | 0.650260 | 0.683909 | ??? | ??? |
| 12-melanoma-5fold-XGBM basic fts external data oof | ??? | 0.664 | ??? | ??? |
| 13-melanoma-5fold-XGBM basic features | ??? | 0.660 | ??? | ??? |
| 14-melanoma-5fold-XGBM random param search | ??? | 0.773 | ??? | ??? |
| 15-melanoma-5fold-XGBM basic features v2 | ??? | 0.681 | ??? | ??? |
| 16-melanoma-5fold-XGBM basic features v3 | ??? | 0.681 | 0.6735 | ??? |
| 17-melanoma-5fold-XGBM basic fts external data v2 | ??? | 0.665 | 0.6840 | ??? |
| 18-melanoma-5fold-XGBM random param search v2 | ??? | 0.783 | 0.7423 | ??? |
| 19-melanoma-stats fts external | ??? | 0.676 | 0.6910 | ??? |
| 20-melanoma-stats fts | ??? | 0.696 | 0.6722 | ??? |
