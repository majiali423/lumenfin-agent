# FinanceBench eval examples

These files document the **schema** of the evaluation harness. They are not the
150-question dataset and must not be treated as gold labels for tuning.

Place the real Patronus FinanceBench checkout (JSONL + `pdfs/`) outside git,
for example under `data/external/`, then run
`scripts/prepare_financebench_eval.py`.

`frozen_config.json` is the machine-readable confirmation-50 lock.
`confirmation_result.json` is the git-tracked aggregate of the recorded
one-shot run (no raw questions, qrels, or per-case rows). Confirmation-50 is
consumed; do not rerun it. Do not treat either file as a license to retune.

License: FinanceBench is CC-BY-NC-4.0. Do not commit PDFs or full evidence
pages.
