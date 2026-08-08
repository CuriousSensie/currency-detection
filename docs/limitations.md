# Limitations and responsible use

## Task boundary

V1 assigns one denomination to an image containing one banknote or banknote crop. It cannot locate notes, separate multiple denominations, count notes, calculate totals, grade condition, or infer authenticity. Visible-light RGB does not expose the security evidence required for banknote authentication.

The classifier is fast enough for live-feed classification experiments on the measured CPU, but only after the frame is constrained to one note or one crop. A live camera application still needs separate capture policy, framing, detection/localization, temporal smoothing, and human-facing error handling.

## Dataset validity

UCP feature boxes are not note boundaries; only their unanimous denomination is used as an image label. Background, capture-device, contributor, design-series, and condition biases may remain. Perceptual-hash grouping reduces obvious leakage but cannot prove semantic independence.

## Experimental uncertainty

The ablation uses one training seed by explicit project decision. Bootstrap confidence intervals cover fixed-test sample variation only. Claims about training variance or statistical significance are unsupported.

## External validity

Abduls crops provide public cross-dataset evidence but have narrow capture conditions. Real-world external validation has not been performed. Public-dataset metrics must not be described as field performance.
