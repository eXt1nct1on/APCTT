# TRL-Scout documentation

Project documentation and runtime prompt references live here.

Run an end-to-end assessment with:

```text
python -m trl_scout assess file.pdf --out out/ [--mock] [--no-cache]
```

The command writes `assessment.json` to the output directory and keeps
per-step audit JSON and `run.log` files under its `data/runs/<run_id>/`
directory.
