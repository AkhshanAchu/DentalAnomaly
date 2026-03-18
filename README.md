# OPG Canine Anomaly Detector

Deep SVDD-based one-class anomaly detection for identifying canine problems
in panoramic dental OPGs. Designed for resource-constrained settings —
the model trains on as few as 10 labelled positive images.

---

## Background

Standard supervised classifiers require hundreds of labelled examples per
class. In clinical practice, collecting balanced datasets is often infeasible.
This project applies Deep Support Vector Data Description (Deep SVDD) to
sidestep that requirement: the model learns a compact hypersphere around
your known positive cases. At inference, any image whose embedding falls
outside that sphere is classified as negative.

    Low anomaly score  (inside sphere)  ->  POSITIVE  (canine problem present)
    High anomaly score (outside sphere) ->  NEGATIVE  (no canine problem)

---

## Results (10-image training set)

Training was run for 150 epochs with 30x augmentation per image (300
effective training samples). A held-out set containing both positive and
negative OPGs was used for threshold calibration.

    ROC-AUC              : 94.76
    Best balanced accuracy: 0.97  at threshold 0.1094
    Anomaly threshold     : 0.1094

Sample inference on the two provided test images:

    IMG_5152.jpeg  ->  POSITIVE  (score well below threshold)
    ren.jpeg       ->  NEGATIVE  (score above threshold)

---

## Architecture

    OPG Image
       |
       v  ROI crop  (anterior canine region: x=[25%, 75%], y=[20%, 80%])
       |
       v  EfficientNet-B0  (ImageNet pretrained, initially frozen)
       |    -> 1280-d feature vector
       |
       v  Projection Head  (1280 -> 512 -> 128 -> 64)
       |    -> compact embedding
       |
       v  Deep SVDD Loss
            -> hypersphere fitted around positive embeddings
            -> anomaly score = L2 distance from hypersphere center

---

## Project Structure

    opg_detector/
    |-- scripts/
    |   |-- train.py          Entry point: training loop
    |   |-- predict.py        Entry point: single image or batch inference
    |
    |-- src/
    |   |-- data/
    |   |   |-- dataset.py    OPGDataset with augmentation factor support
    |   |   |-- transforms.py Train and eval image transforms
    |   |   |-- roi.py        Canine ROI crop logic
    |   |
    |   |-- model/
    |   |   |-- backbone.py   EfficientNet-B0 feature extractor
    |   |   |-- head.py       MLP projection head
    |   |   |-- svdd.py       SVDDLoss and center initialisation
    |   |   |-- detector.py   CanineAnomalyModel (full pipeline)
    |   |
    |   |-- engine/
    |       |-- trainer.py    Single epoch training step
    |       |-- evaluator.py  Evaluation loop and threshold calibration
    |       |-- predictor.py  Model loading and image inference
    |
    |-- data/
    |   |-- positive/         OPGs with confirmed canine problem (training)
    |   |-- negative/         Normal OPGs (optional; used for calibration)
    |
    |-- checkpoints/          Saved best_model.pt and config.json
    |-- requirements.txt
    |-- README.md

---

## Setup

    pip install -r requirements.txt

---

## Data Preparation

Place your images in the following layout:

    data/
        positive/    <-- OPGs WITH the canine problem  (your ~10 images)
        negative/    <-- normal OPGs  (optional; improves threshold calibration)

Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif

---

## Training

    python scripts/train.py \
        --data_root ./data \
        --output    ./checkpoints \
        --epochs    150 \
        --augment   30

Key arguments:

    --augment   INT    Virtual copies per image via augmentation  (default: 30)
    --warmup    INT    Frozen-backbone epochs before fine-tuning  (default: 10)
    --nu        FLOAT  Fraction of allowed outliers in SVDD       (default: 0.1)
    --embed_dim INT    Hypersphere dimensionality                 (default: 64)
    --no_roi           Disable automatic canine ROI crop

On completion, checkpoints/ will contain:

    best_model.pt   PyTorch model weights and hypersphere center
    config.json     Threshold and inference configuration

---

## Inference

Single image:

    python scripts/predict.py \
        --image      /path/to/opg.jpg \
        --checkpoint ./checkpoints

Batch (folder):

    python scripts/predict.py \
        --folder     /path/to/test_opgs/ \
        --checkpoint ./checkpoints

Example output:

    ------------------------------------------------------------
    File:       patient_07.jpg
    Score:      0.0821  (threshold=0.1094)
    Prediction: POSITIVE -- Canine problem detected
    Confidence: High confidence (well inside hypersphere)

---

## Design Decisions for Low-Data Regimes

    One-class learning      No negative examples required during training.
                            Negatives are only used for threshold calibration.

    Pretrained backbone     EfficientNet-B0 ImageNet weights transfer
                            effectively to grayscale medical imagery
                            when input is replicated to 3 channels.

    Heavy augmentation      30x virtual expansion (flips, rotations, affine,
                            brightness jitter) yields 300 effective samples
                            from 10 originals.

    Canine ROI crop         Focuses the model on the clinically relevant
                            anterior region, reducing background noise.

    Soft-boundary SVDD      Radius R is learnable, accommodating label noise
                            and intra-class variation within small datasets.

    Two-phase training      Backbone frozen during warmup to stabilise
                            the hypersphere center; unfrozen at 0.1x LR
                            for fine-tuning on domain features.

---

## Tuning Tips

- Add 5-10 normal OPGs to data/negative/ for more reliable threshold calibration.
- Adjust ROI_X_START / ROI_X_END / ROI_Y_START / ROI_Y_END in src/data/roi.py
  if canines are positioned differently in your scanner's output.
- Try --embed_dim 32 for a tighter sphere or 128 for more flexibility.
- If dental-specific pretrained weights are available (e.g. RadImageNet),
  substitute them in src/model/backbone.py for improved feature quality.

---

Made with ❤️ by NiceGuy
