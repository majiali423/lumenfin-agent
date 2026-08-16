# FinanceBench eval examples

These files document the **schema** of the evaluation harness. They are not the
150-question dataset and must not be treated as gold labels for tuning.

Place the real Patronus FinanceBench checkout (JSONL + `pdfs/`) outside git,
for example under `data/external/`, then run
`scripts/prepare_financebench_eval.py`.

`frozen_config.json` is the machine-readable confirmation-50 lock. Do not treat
it as a license to run confirmation without `--confirm-held-out`.

License: FinanceBench is CC-BY-NC-4.0. Do not commit PDFs or full evidence
pages.
