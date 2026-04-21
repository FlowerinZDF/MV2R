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

