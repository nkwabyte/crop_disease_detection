# Adopting the 4-class classifier in the mobile app

A decision record for whether to replace the shipped 3-class crop classifier with
the 4-class variant that has a learned `Other` class.

**Recommendation: adopt it — but as a migration, not a file swap, and with a
confidence floor kept on top.** The reasoning, the evidence, the work it implies
and the conditions that would change this answer are all below.

Measured 2026-08-04 on an RTX 5090. Raw numbers:
[`outputs/benchmarks/classifier_variant_comparison.json`](../outputs/benchmarks/classifier_variant_comparison.json)
· charts in `outputs/benchmarks/figures/`.

---

## 1. The problem being solved

The shipped classifier is a 3-class softmax (Corn / Pepper / Tomato) with no way
to say *"this is not a crop"*. Softmax always sums to 1, so a photo of a car, a
hand, or a cassava leaf is forced into one of the three classes. Rejection is a
single threshold applied afterwards:

```python
if conf < CONF_DEFAULT:      # 0.55
    return "unknown", conf, []
```

`CONF_DEFAULT = 0.55` was never measured. It turns out to be far too permissive:
**at 0.55 the model rejects only 40.8 % of non-crop images** — nearly six in ten
get through and are handed to the detector as a real crop.

This also means the claim in `src/app/app_gradio.py` that a mango leaf scores
*"unknown — 23 %"* and is rejected was never verified and overstates the guard.

## 2. The two options measured

Both trained on identical crop data and scored on the identical test set —
1,016 crop images and 475 non-crop images.

| Model | Crop accuracy | Non-crop rejection |
| ----- | ------------- | ------------------ |
| 3-class, no threshold | 97.54 % | 0 % |
| 3-class, threshold 0.55 *(shipped)* | 97.44 % | **40.84 %** |
| 3-class, threshold 0.90 | 82.78 % | 94.74 % |
| **4-class, learned `Other`** | **97.54 %** | **98.74 %** |

The 4-class model matches the *unthresholded* crop accuracy exactly while
rejecting 98.7 % of non-crop images. Thresholding cannot reach that: buying
94.7 % rejection costs 15 points of crop recall, and 0.95 collapses retention to
10.6 %. Only **0.39 %** of genuine crops are misrouted to `Other`.

Cost on device is effectively nil — one extra output neuron on the same
EfficientNet-B2 backbone. Both `.pte` files are 29 MB; latency is unchanged.

### Rejection by source — read this before trusting the headline

| Source | 3-class @ 0.90 | 4-class | n (test) |
| ------ | -------------- | ------- | -------- |
| **eggplant** | 74.47 % | **93.62 %** | 47 |
| potato | 94.19 % | 97.67 % | 86 |
| millet | 95.00 % | 100 % | 20 |
| tobacco | 95.65 % | 100 % | 23 |
| sorghum | 80.00 % | 100 % | 5 |
| generic photographs | 98.30 % | 99.66 % | 294 |

Eggplant is hardest for both models, which is the expected result: it is
*Solanaceae*, the same family as pepper and tomato. **93.6 % is the honest
estimate of near-miss performance**, not the 98.7 % headline — the headline is
lifted by the 294 generic photographs, which are an easy case.

## 3. What adopting it changes in the code

This is **not a drop-in replacement**. The `.pte` emits 4 logits instead of 3 and
rejects by `argmax` rather than by threshold, so model and inference code must
change together. Concrete sites:

| Site | Change |
| ---- | ------ |
| `src/classifier/train_classifier.py` · `CropClassifier.predict()` | `CROP_TO_YOLO_CLASSES[label]` raises **KeyError** on `"Other"` — must route to the unknown path first |
| same · rejection branch | `argmax == 3` instead of `conf < self.threshold` |
| `src/classifier/config.py` | `CROP_CLASSES` / `NUM_CLASSES` gain a 4th entry for the app path |
| `src/app/app_gradio.py` | confidence slider ("images below this score are rejected") no longer drives rejection — rework or remove |
| `src/app/app_gradio.py` line ~738 | the documented "Mango leaf → unknown 23 %" example is wrong either way; re-measure |
| Android inference | reads 4 logits; rejection by argmax |
| `models/*_metadata.yaml` | already records `class_names` and the `rejection` rule per artifact — read it rather than hard-coding |

Estimated effort: roughly half a day, but it has to land **atomically** with the
new `.pte`. Shipping the model without the code change produces a KeyError on the
first non-crop photo.

## 4. Recommended configuration

Ship the 4-class model **and keep a confidence floor on the three crop classes.**

```python
probs = softmax(logits)          # 4 logits
idx, conf = argmax(probs), max(probs)
if idx == OTHER or conf < CONF_FLOOR:
    return "unknown"
return CROP_CLASSES[idx]
```

The learned class catches what it was trained on; the floor still catches
confidently-wrong predictions on species it has never seen. The two mechanisms
fail differently, which is the point — and the floor costs nothing, since at
0.55 it barely touches genuine crops (97.44 % retention).

Keep `crop_classifier.pte` shipping alongside `crop_classifier_ood.pte` so the
two can be A/B'd on real user photos before the old one is retired. Both already
export side by side — see [`scripts/export/README.md`](../scripts/export/README.md).

## 5. Risks and what they cost

| Risk | Assessment |
| ---- | ---------- |
| **Rejection will be worse in the field than 98.7 %** | Near-certain. The negatives cover 5 species; a Ghanaian farmer will photograph cassava, plantain, cocoa, yam, okra and weeds that the model has never seen. Plan around ~93 %, the eggplant figure. |
| **Small test n** | 475 negatives; eggplant 47, sorghum 5. Per-source rates are indicative, not precise. Sorghum's 100 % rests on 5 images. |
| **New failure mode: valid crops rejected** | 0.39 % of genuine crops are called `Other`. Rare, but a worse user experience than a wrong crop guess — a wrong guess still yields a diagnosis, a rejection is a dead end on a legitimate leaf. |
| **`Other` is a discriminative class, not an OOD detector** | It has learned "things that look like the negatives we showed it". It gives no calibrated guarantee on genuinely novel input. The confidence floor is the mitigation. |
| **Coupling to the negative set** | If a negative species later becomes a target crop, its folder must be removed from `data/negatives/` and the model retrained. The tooling supports this by design (§6). |

## 6. If a negative later becomes a target crop

The negatives are stored one folder per source specifically for this:

```bash
rm -rf data/negatives/eggplant                       # promote eggplant to a crop
python -m src.classifier.generate_classifier_csv --with-negatives
bash scripts/train_classifier.sh --variant ood
python -m src.classifier.compare_variants
```

`generate_classifier_csv.py --with-negatives` enumerates whatever subdirectories
exist, so removing a folder is the entire workflow.

## 7. Before relying on this in the field

In rough priority order:

1. **Add cassava, plantain and cocoa to the negatives.** They are the three most
   likely things a Ghanaian farmer points a phone at that are not covered. Field
   photographs only — see the warning below.
2. **Re-measure the app's rejection claims** and correct the documented examples.
3. **A/B on real user photos** before retiring the 3-class model.
4. **Grow the eggplant test set** beyond 47 images; it is the binding case and
   currently the least certain number.

> **Sourcing warning.** PlantVillage and Project-AgML were both evaluated as
> negative sources and **rejected**: their images are detached leaves on plain
> studio backgrounds, whereas this project's images are field photographs. A
> model trained against them learns *"plain background → Other"*, which scores
> beautifully in validation and does nothing for the real failure. Use field
> photographs — leaf on the plant, natural background, phone camera. The five
> species used here all satisfy that.

## 8. What would change this recommendation

- Near-miss rejection measured **below ~85 %** on a broader species set — the
  gain would no longer justify the migration.
- Crop rejections (`Other` on a genuine leaf) measured **above ~2 %** in the
  field — the dead-end UX would start to outweigh the benefit.
- The app moving to a design where a wrong crop guess is cheap (e.g. the user
  picks the crop manually), which would remove most of the value.

## Related

- [`outputs/benchmarks/classifier_variant_comparison.md`](../outputs/benchmarks/classifier_variant_comparison.md) — the results tables
- [`scripts/export/README.md`](../scripts/export/README.md) — producing both `.pte` artifacts
- [`docs/01_crop_classifier.md`](01_crop_classifier.md) — the classifier itself
- [`docs/06_two_stage_pipeline.md`](06_two_stage_pipeline.md) — how the classifier feeds the detector
