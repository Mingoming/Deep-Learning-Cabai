# Project instructions

- Keep Klasifikasi_Daun_Cabai.ipynb as the single canonical training workflow; avoid unnecessary application modules.
- Keep one-time extraction, image audit, deduplication, and manifest creation in scripts/prepare_data.py. The notebook reads verified artifacts and does not repeat preparation when they already exist.
- Keep notebook cells linear and readable for a student audience; training must remain an explicit final action rather than running on notebook startup.
- Treat docs/DEEP LEARNING.md, docs/DEEP LEARNING PPT.pdf, and the original ZIP in data/ as read-only sources.
- The previous-year notebook is a presentation reference only. Do not copy its methodology, architectures, or hyperparameters.
- Keep extracted images and generated audit/manifests under data/; never stage or commit dataset files.
- Follow the owner-approved decisions in docs/DL-01-DL-02-decisions.md: PPT takes precedence over the older report for experiment design; use the existing group-aware manifest, seeds 42/43/44, and train-only augmentation and class weights.
- Reserve test data for final evaluation. Do not run full training, other architectures, or CBAM without separate authorization; a bounded smoke check is allowed when the runtime is available.
- After data-pipeline changes, run .venv/Scripts/python.exe -m unittest discover -s tests -v.
- Keep model code and evaluation in the canonical notebook. Use scripts/prepare_data.py only for one-time dataset preparation; do not add other application modules unless requested.
