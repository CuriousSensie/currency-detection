# Classification dataset card

## Status and intended use

Canonical construction completed from locally verified archives on 2026-09-10. The dataset supports seven-class, single-banknote image classification. It does not support whole-note localization, multiple-note counting, damage assessment, or authenticity claims.

## Sources

| Source | License | Role | Observed outcome |
|---|---|---|---|
| UCP Pakistani Currency Note Data v1 | CC BY 4.0 | Internal training, validation, and test | 4,006 images; 3,565 eligible and 441 PKR 75 exclusions |
| Abduls PKR_Notes v1 | Public Domain | Cross-dataset crop evaluation only | 169 images; 168 structurally valid, yielding 327 valid note crops |
| Copperhorse/CurrencyClassification | No formal repository license | Related work only | No code, weights, data, or claimed metrics imported |

UCP’s boxes identify denomination/security regions, not whole-banknote extents. All 3,565 eligible images have unanimous annotation classes, so those classes are used only as image-level labels. The images are letterboxed to 224×224 without another aspect-ratio distortion.

Abduls boxes surround whole notes and may contain several notes per source image. Every crop records its source image, original box, source and derived hashes, and denomination. One source image is rejected because a polygon row appears in a detection label file.

## Leakage controls and splits

SHA-256 groups byte-identical inputs. A 64-bit difference hash with maximum Hamming distance four groups likely near duplicates before the deterministic stratified split. Cross-source perceptual matches of the same denomination are excluded from Abduls evaluation.

| Internal split | Images |
|---|---:|
| Train | 2,496 |
| Validation | 535 |
| Test | 534 |

No ambiguous UCP image labels or cross-source overlaps were observed. The external partition contains 327 crops. The canonical manifest SHA-256 is `d36533b491482a3eb4b84cdf6739ab346ac6f5d619fc59e088eca4d0e0e08413`.

## Limitations

UCP images may contain background correlations, partial notes, blur, and source-specific capture patterns. Difference hashing is an imperfect proxy for scene identity. Abduls is small, visually narrow, and public; it is useful for domain-shift evidence but is not independently collected field validation. Contributor and camera identities are unavailable, preventing ideal subject/device grouping.

The smallest training class has 351 images. The 10, 50, 100, and full-data ablations are feasible; 500/class is retained as an explicit omitted condition.
