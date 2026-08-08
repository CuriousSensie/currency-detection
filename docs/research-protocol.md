# Pre-registered classification protocol

## Questions

1. How does denomination classification change at 10, 50, 100, and all available training images?
2. Under matched initialization, membership, optimization, and seed, what changes when the registered augmentation recipe is enabled?
3. How well does the validation-selected classifier transfer from UCP images to isolated Abduls note crops?

## Fixed design

- Seed: `20260908`; architecture: ImageNet-pretrained EfficientNet-B0; input: 224×224 letterboxed RGB.
- Classes: PKR 10, 20, 50, 100, 500, 1000, and 5000.
- Internal data: UCP v1, grouped 70/15/15 split after PKR 75 exclusion and duplicate detection.
- External data: Abduls ground-truth note crops, screened against all UCP images and never used for fitting or selection.
- Conditions: no augmentation and the fixed plausible augmentation recipe.
- The 500/class condition stays in the ledger as omitted because the smallest training class has 351 images.

## Selection and calibration

Each run selects its checkpoint by validation macro F1. The two full-data conditions are compared using validation macro F1; within 0.002, lower validation ECE wins, then no augmentation on an exact tie. Temperature scaling fits validation logits after checkpoint selection. Test and external labels make no training, selection, or calibration decision.

## Reporting

Report accuracy, balanced accuracy, macro F1, per-class precision/recall/F1, confusion matrices, ECE, Brier score, top-2 accuracy, and 2,000-iteration stratified bootstrap intervals. Intervals describe fixed-test sampling uncertainty and exclude training-seed variance. Any deviation is recorded before interpretation; missing and failed runs remain explicit.

## Product gate

The model serves only at internal macro F1 ≥0.90, external macro F1 ≥0.70, and internal recall ≥0.75 for every class. Results below the gate remain valid research evidence but do not enable the product.
