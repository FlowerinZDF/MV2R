# MV2R Roadmap

## Project Goal
MV2R extends the LVLM4FV-style baseline toward view-level reasoning for multimodal fake news detection.

## Development Principle
- Keep the original baseline pipeline as intact as possible.
- Implement new MV2R components under `src/`.
- Avoid invasive refactoring in the baseline code during the first stage.

## Stage 1 Goal
Build a minimal runnable MV2R prototype that supports:
- overall label prediction
- fixed view-level label prediction

## Fixed Views
We use the following fixed views in the first version:
- subject
- scene
- event
- time
- location
- ocr
- relation
- tampering

## New Modules
- `src/data/schema.py`: sample schema
- `src/data/dataset.py`: dataset loader
- `src/data/collator.py`: batch collation
- `src/models/view_head.py`: view-level prediction head
- `src/models/aggregator.py`: overall aggregation head
- `src/llm/view_decomposer.py`: lightweight view decomposition interface
- `src/trainers/train_mv2r.py`: minimal training script
- `src/eval/eval_overall.py`: overall evaluation
- `src/eval/eval_view.py`: view-level evaluation

## Stage 1 Non-Goals
The following are not required in the first runnable version:
- fine-grained image region grounding
- complex multi-agent reasoning
- full explanation generation
- end-to-end joint optimization of all modules

## Immediate Next Steps
1. Define the MV2R sample schema.
2. Implement a minimal dataset loader.
3. Prepare a small pilot dataset.
4. Add a minimal multi-view prediction head.
5. Run the first overall + view-level experiment.
## Pilot Dataset
- Added initial pilot file: `data/pilot/mv2r_pilot.json` (20 schema-compliant samples for first real MV2R experiment).

- Split pilot into train/val: `data/pilot/mv2r_pilot_train.json` (16) and `data/pilot/mv2r_pilot_val.json` (4) for first-pass evaluation.

## Stable Pilot Round (v2)
- Added larger pilot split for next stable round:
  - `data/pilot/mv2r_pilot_train_v2.json` (48)
  - `data/pilot/mv2r_pilot_val_v2.json` (12)
- Added combined mock image-feature file for multimodal integration testing:
  - `data/pilot/mv2r_pilot_v2_image_features.json` (train+val coverage, deterministic 16-dim vectors keyed by sample ID)
- Current best-performing configuration:
  - improved shared encoder
  - `view_loss_weight = 1.0`
  - `aggregator_mode = shared_and_view_logits`

## Experimental Note (Shortcut Risk)
- `conflict_type` is now treated as analysis metadata rather than default model input in the prototype baseline.
- Pilot ablations indicated `conflict_type` can act as a shortcut signal and introduce leakage-like behavior in validation.

## Pilot v2 Experiment Summary
- Current trusted baseline (default safe):
  - `encoder_type = multimodal_ready_text`
  - `aggregator_mode = shared_and_view_logits`
  - `view_loss_weight = 1.0`
  - `conflict_type` disabled by default
- Why `conflict_type` is disabled by default:
  - It can become a shortcut signal and inflate validation behavior without improving robust generalization.
- Weak image-hint result:
  - `multimodal_light` and `multimodal_image_features` (mock/weak image hints) did not outperform the trusted baseline.
- Recommendation for future multimodal work:
  - Prefer richer semantic image embeddings over weak mock feature vectors for next-stage multimodal experiments.

## Pilot v2 Findings (Compact)
| Configuration | Observation |
| --- | --- |
| `simple_text` + `shared_and_view_logits` | Useful reference point but not the trusted default after v2. |
| `multimodal_ready_text` + `shared_and_view_logits` + `conflict_type` | Can look strong but depends on a shortcut-prone signal. |
| `multimodal_ready_text` + `shared_and_view_logits` (default safe) | Current trusted pilot baseline. |
| `multimodal_light` + `shared_and_view_logits` | Weak image hints did not beat trusted baseline. |
| `multimodal_image_features` + `shared_and_view_logits` | Mock image features did not beat trusted baseline. |
