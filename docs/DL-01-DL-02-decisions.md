# DL-01 closure and DL-02 baseline decisions

This note records the project-owner decisions for the baseline preparation. The Progress I report, PPT, and original dataset ZIP remain unchanged.

## Methodology conflicts and resolution

The local Progress I report describes per-image stratified splitting at 70:15:15 and pixel rescaling to [0,1] (section 3.1, paragraphs beginning “Sebelum dibagi” and “Seluruh citra diubah ukurannya”). It also says “Model dilatih dari awal (training from scratch)” and chooses the best model by “validation loss terendah” (section 3.1, paragraphs beginning “Model dilatih” and “Skenario eksperimen”).

The local Progress I PPT differs: slide 9 describes the source images as 1000×1000 RGB and specifies ImageNet normalization; slide 10 lists MobileNetV3-Small among the transfer-learning configurations and describes “Transfer learning 2 tahap: classification head → fine-tuning seluruh jaringan”, AdamW, label smoothing 0.1, and cosine annealing.

Project-owner decision: where these designs conflict, follow the PPT. Use ImageNet-pretrained MobileNetV3-Small, RGB input resized to 224×224, and the preprocessing mean/std from the selected TorchVision weights. Use the head-only then whole-network fine-tuning sequence. The owner also selected validation macro-F1 as the primary metric and checkpoint-selection metric. This supersedes the report's validation-loss selection wording; validation loss remains a diagnostic.

The owner approved retaining the existing 70:15:15 group-aware split, exact-duplicate exclusions, near-duplicate groups, and manifests for seeds 42, 43, and 44. This is an intentional departure from per-image stratification to keep candidate-correlated images in one subset. Test remains evaluation-only.

## Existing audit and manifest evidence

The figures below are read from the already validated files under data/processed. The ZIP was not re-extracted or re-audited during this closure.

- 8,817 image files were valid and decodable; none were recorded as corrupt.
- The source archive size is 3,037,394,349 bytes and its audited SHA-256 is 7b95cbbf3b725d3fc7f7315438efb374c5ce58e226473975a28e5079f7a08c4d.
- The archive contained 2,793 JPEG/RGB and 6,024 PNG/RGBA images across 1,968 distinct resolutions; only 2,028 files were 1000×1000. The loader converts images to RGB and resizes them to the approved 224×224 input.
- Exact deduplication leaves 7,936 unique images, 881 fewer than the valid raw files; cross-label exact copies are marked ambiguous and excluded.
- Raw-valid and exact-unique image counts, followed by the seed-42 split counts, were:
- The 64-bit dHash audit found 2,712 candidate pairs at Hamming distance ≤4. The saved review covered 13 cross-class candidate pairs; no clear same-photo copies were identified in those reviewed pairs. Labels were not changed from this review.
- 642 non-singleton near-duplicate groups cover 1,877 images. The previously verified manifests for seeds 42/43/44 have no exact-hash or near-duplicate-group assignments crossing train, validation, and test, and recomputation against the notebook assignment function produced no mismatches.
- Each seed has train 5,556, validation 1,191, and test 1,189 images. Per-class train/validation/test counts are the same for all three seeds:

| Class | Raw valid | Exact unique | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Bacterial Spot | 1,629 | 1,280 | 896 | 192 | 192 |
| Cercospora Leaf Spot | 1,898 | 1,683 | 1,178 | 253 | 252 |
| Curl Virus | 1,590 | 1,582 | 1,108 | 237 | 237 |
| Healthy Leaf | 1,647 | 1,481 | 1,037 | 222 | 222 |
| Nutrient Deficiency | 1,207 | 1,067 | 747 | 160 | 160 |
| Powdery Mildew | 846 | 843 | 590 | 127 | 126 |

These counts describe the audited dataset and split, not model performance.

## Limitations

dHash is a screening heuristic. Visual inspection of candidate pairs cannot establish that images came from different leaves or plants, and the dataset has no plant/leaf identity metadata. Grouping can both fail to connect correlated photos and conservatively combine visually similar but independent images. The group-aware method also cannot promise exact 70:15:15 ratios for every class; the saved counts above are the actual result. Cross-label exact duplicates remain excluded as ambiguous in the existing manifest; their labels were not edited.

The report and PPT both state 8,814 images, while the supplied ZIP audit records 8,817 valid image files. The notebook preserves the ZIP-derived count; this source-count discrepancy is documented and the source documents were not edited.

## Baseline proposal

The source-supported settings are AdamW, label smoothing 0.1, cosine annealing, ImageNet initialization, and the two-stage schedule. The following starting values are an implementation proposal, not values reported in Progress I and not experiment results:

- Pilot seed 42, batch size 32.
- Head-only stage: 3 epochs at learning rate 1e-3.
- Whole-network fine-tuning: 8 epochs at learning rate 1e-4.
- AdamW weight decay 1e-4; cosine annealing per stage with eta_min 1e-6.
- Weighted cross-entropy with inverse-frequency weights computed from seed-42 training rows only.
- Keep the existing train-only horizontal flip (p=0.5), rotation (±15°), and brightness/contrast jitter (0.2 each) as provisional magnitudes.

The short smoke run is capped at two train batches and two validation batches per stage and one epoch per stage. It checks plumbing only. Its metrics and checkpoints must not be treated as model-selection evidence.

## Runtime and official API references

The previously recorded WSL2 enumeration returned Wsl/EnumerateDistros/Service/E_ACCESSDENIED before any distro or GPU could be inspected. The prepared Colab instructions remain in the canonical notebook, but the browser runtime could not be opened in this session. No policy was changed or bypassed, and no training smoke was executed.

The notebook targets the pinned PyTorch 2.10 / TorchVision 0.25 APIs. References: [TorchVision 0.25 API index](https://docs.pytorch.org/vision/0.25/genindex.html), [MobileNetV3-Small model API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html), [AdamW](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html), and [CosineAnnealingLR](https://docs.pytorch.org/docs/2.10/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html). The stable TorchVision model page is newer than the pinned 0.25 package; its API was cross-checked against the locally installed TorchVision 0.25.0 source.
