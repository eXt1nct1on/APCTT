# TRL-Scout documentation

Project documentation and runtime prompt references live here.

Run an end-to-end assessment with:

```text
python -m trl_scout assess file.pdf --out out/ [--mock] [--no-cache]
```

The command writes `assessment.json` to the output directory and keeps
per-step audit JSON and `run.log` files under its `data/runs/<run_id>/`
directory.

Build a report from an assessment:

```text
python -m trl_scout.report.builder docs/example_assessment.json out/report.pdf
```

The report is a self-contained HTML document with embedded charts. When
WeasyPrint is installed, the same command also writes the requested PDF;
otherwise it logs a warning and leaves the HTML beside the requested path.
