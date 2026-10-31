# Gold evaluation set

`labels.csv` is the human-labelled evaluation manifest. Each row identifies a
document and records the final TRL, uncertainty range, two independent labels,
and notes explaining difficult decisions.

## Labelling protocol

1. Read the document and use the authoritative rubric in
   `src/trl_scout/data/rubric.yaml`; do not infer a level from an author's
   explicit TRL claim alone.
2. Record achieved evidence only when the document demonstrates the criterion.
   Plans, hypotheses, simulations, and cited work must be marked accordingly.
3. Two labelers independently record `labeler_a` and `labeler_b`, then discuss
   any disagreement greater than one TRL level.
4. Record the agreed final value in `gold_trl`, with `gold_low` and `gold_high`
   for genuine uncertainty. Explain the decision in `notes`.
5. Keep `doc_id` stable and use a local filename under `eval/gold/docs/`.

The three initial rows are clearly invented examples for smoke testing and are
not scientific ground truth.
