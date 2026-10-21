# CheapMage - Human Evaluation Protocol & Score Sheet

This is the **only** accepted way to score image quality. Automated metrics
(latency, cost, similarity) can tell us that something is fast, cheap and
self-consistent; they cannot tell us that it does not look like AI slop.
Every release ships a filled-in copy of this protocol in `docs/REPORT.md`.

- Protocol version: **1.0**
- Applies to: `cheapmage-v1` and every candidate that competes with it
- Scales: **1 - 5**, integer only (no half points - half points are where
  raters start negotiating instead of scoring)

---

## 1. What we evaluate

Three blocks, each scored separately. Every image receives all three blocks.

| Block | Question | Higher is better? |
|---|---|---|
| **A - Anime quality** | Does this look like a usable anime/2D character illustration? | yes |
| **B - 3D quality** | Does this look like a usable stylised 3D character render? | yes |
| **C - AI slop** | How much "AI slop" is visible? | **no** (5 = worst) |

Block C is scored **inverted on purpose**: "5" means *severe slop*, "1" means
*clean*. Reporting it this way makes it impossible to accidentally average a
slop score into a quality score.

An image is only scored in the block matching its `style_preset`:

- `style_preset = anime` -> block **A**
- `style_preset = 3d` -> block **B**
- `style_preset = auto` -> the rater classifies it first and picks the block

Every image is additionally scored for block **C** (slop applies to both
styles).

---

## 2. Rating scale

### Quality blocks (A and B)

| Score | Label | Meaning |
|---|---|---|
| 5 | Excellent | Ship-ready. Would be used as an avatar/scene image without edits. |
| 4 | Good | Clearly usable. Small blemishes a user would not mention. |
| 3 | Acceptable | Usable with minor doubt. One visible flaw. |
| 2 | Poor | Recognisable character, but a flaw that a user would notice. |
| 1 | Unusable | Broken: wrong anatomy, unreadable face, heavy artefacts. |

### Slop block (C) - inverted

| Score | Label | Meaning |
|---|---|---|
| 1 | No slop | Nothing that reads as "generated". |
| 2 | Mild slop | One giveaway, easy to overlook at a glance. |
| 3 | Moderate slop | Two or three giveaways; a careful viewer notices. |
| 4 | Heavy slop | Slop is the first thing you see. |
| 5 | Extreme slop | Looks broken, not stylised. |

---

## 3. Rubrics

### Block A - Anime quality

| Criterion | What "5" looks like | What "1" looks like |
|---|---|---|
| Line quality | Clean, confident, consistent line weight; no breaks | Broken, scratchy, doubled or missing lines |
| Face readability | Eyes, nose and mouth are clearly readable and symmetric | Cross-eyed, melted, asymmetric or missing features |
| Hair | Coherent strand grouping, consistent volume, believable silhouette | Melted, fused into a helmet, or strands ending in nothing |
| Body proportions | Believable anime proportions, no stray limbs | Extra/missing limbs, broken joints, impossible anatomy |
| Colour/cel shading | Flat or two-tone shading that follows the form, clean palette | Muddy, randomly shaded, palette broken between regions |
| Composition | Character well framed, readable against the background | Cropped head, out of frame, merges into the background |

### Block B - 3D quality

| Criterion | What "5" looks like | What "1" looks like |
|---|---|---|
| Silhouette & form | Volumetric, believable 3D form with coherent perspective | Flat or collapsed shapes, impossible geometry |
| Material / skin | Believable subsurface look, no uniform plastic sheen | Uniform wax/plastic sheen, or noisy fake skin |
| Shading & lighting | Single coherent light direction, soft falloff, plausible contact shadows | Multiple conflicting light directions, hard-edged "cut-out" shadows |
| Render cleanliness | Clean edges, no noise, no smearing | Fireflies, banding, smeared textures, background leakage |
| Anatomy | Correct proportions, believable joints and hands | Deformed hands, broken joints, impossible anatomy |
| Composition | Character well framed with clean studio background | Cropped, out of frame, clipped by the frame edge |

### Block C - AI slop criteria (scored 1 = none, 5 = severe)

| # | Criterion | 1 | 3 | 5 |
|---|---|---|---|---|
| C1 | **Plastic skin** | Skin has natural variation | Slight uniform sheen | Uniform wax/plastic, no skin detail at all |
| C2 | **Malformed hands** | Hands anatomically correct | One odd finger or blurry hand | Extra/missing/fused fingers, melted hands |
| C3 | **Malformed eyes** | Eyes symmetric and correctly placed | Slight asymmetry, dull iris | Cross-eyed, mismatched, dead or duplicated eyes |
| C4 | **Detail noise** | Clean surfaces, coherent detail | Some speckling in flat areas | Speckle/boiling texture, random detail that fights the subject |
| C5 | **Lighting consistency** | One coherent light source throughout | Soft inconsistency between regions | Different light direction per body part |
| C6 | **Palette consistency** | Cohesive palette across the frame | One clashing region | Random per-region colours, oversaturated neon |
| C7 | **Duplicated anatomy** | Exactly one character, no repeats | A small odd repetition | Extra head/limb, duplicated character |

**Slop sub-score** = mean of C1..C7 (one decimal). The block C score reported
in `docs/REPORT.md` is this mean.

---

## 4. The 30-image set

The set is fixed and reproducible:

- 10 anime images (`style_preset = anime`)
- 10 3D images (`style_preset = 3d`)
- 10 auto images (the engine picks the style from the prompt)

Generation command (writes one JSON manifest that names every file):

```
python -m cheapmage.bench.bench_human_eval --count 30 --out reports/human_eval_set.json
```

If that script is not available yet, the equivalent is: render the prompts in
`cheapmage/bench/human_eval_prompts.json` with `bench_speed --keep-images`
and record the file names in the manifest.

The 30 prompts are chosen to cover the hard cases on purpose: hands in frame,
close-up faces, full body, action pose, seated pose, and two back-lit /
high-contrast setups (lighting consistency is where slop hides).

## 5. Rater procedure

1. **Three raters** score every image independently. Raters must not discuss
   an image before both have submitted their score for it.
2. **Blinding**: images are presented in a random order with no file names, no
   prompt text and no style label. The rater picks the block (A or B) from the
   image alone.
3. **Scale display**: the image is shown at 512x512 (the native output size)
   and may be zoomed to 100%. Scores must be given at 512x512; zooming is only
   allowed to inspect a suspected defect, and the score must reflect the
   512x512 view.
4. **Time box**: at most 45 seconds per image. Raters that linger are biased
   towards over-scoring detail noise.
5. **One pass, no back-tracking**: the rater may not change a submitted score.
   If they want to change it, they must write a note and a second rater acts
   as tie-breaker.
6. **Notes**: any image scoring <= 2 in a quality block or >= 4 in the slop
   block must get a one-line free-text note naming the defect
   (e.g. "C2: five fingers on the right hand").

### Scoring sheet (one row per image per rater)

| Field | Values |
|---|---|
| `image_id` | `A01`..`A10` (anime), `B01`..`B10` (3D), `C01`..`C10` (auto) |
| `rater` | `R1`, `R2`, `R3` |
| `style_block` | `A` or `B` (rater's own classification) |
| `quality_score` | 1..5 |
| `slop_mean` | mean of C1..C7, one decimal |
| `slop_flags` | which of C1..C7 scored >= 3 (comma separated) |
| `note` | free text (required for quality <= 2 or slop >= 4) |

## 6. Aggregation

For every image and every block:

```
per_image_quality  = median(R1.quality_score, R2.quality_score, R3.quality_score)
per_image_slop      = median(R1.slop_mean,    R2.slop_mean,    R3.slop_mean)
```

The **median** is used (not the mean) so a single distracted rater cannot
move a release verdict; the spread is reported as well:

```
rater_spread = max(R1, R2, R3) - min(R1, R2, R3)
```

Release level numbers:

```
anime_quality_mean = mean(per_image_quality for the 10 anime images)
3d_quality_mean    = mean(per_image_quality for the 10 3D images)
slop_mean          = mean(per_image_slop over all 30 images)
```

A rater is considered *out of calibration* when their mean quality score
deviates from the panel mean by more than 0.75 points; such a rater's scores
are kept but flagged in `docs/REPORT.md`.

### Acceptance thresholds

| Metric | Threshold | Hard fail? |
|---|---|---|
| `anime_quality_mean` | >= 3.8 / 5 | yes |
| `3d_quality_mean` | >= 3.6 / 5 | yes |
| `slop_mean` | <= 1.8 / 5 | yes |
| Anime images scored <= 2 by the panel | <= 1 of 10 | yes |
| 3D images scored <= 2 by the panel | <= 1 of 10 | yes |
| Images with a C2 (hands) or C3 (eyes) flag | <= 2 of 30 | yes |
| `rater_spread` per image | <= 2 points for >= 27 of 30 images | no (flagged) |

## 7. Output

The aggregated result is written to `reports/human_eval.json` and summarised
in `docs/REPORT.md` under "Human evaluation". The raw per-rater sheet is kept
in `reports/human_eval_raw.csv` so the aggregation can be re-checked.

## 8. Current status

**TODO - pending run** - no human evaluation has been run yet.

Command that produces the numbers:

```
python -m cheapmage.bench.bench_human_eval --count 30 --out reports/human_eval.json
```
