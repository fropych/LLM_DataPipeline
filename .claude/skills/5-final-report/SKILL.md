---
name: 5-final-report
description: Compile final README (5 sections), data card, run_pipeline.py
allowed-tools: Bash, Read, Write, Glob, Grep
---

Compile the final project deliverables.

1. Read all reports from `reports/` directory
2. Write `README.md` with 5 required sections:
   - **Section 1 — Task and dataset**: modality, volume, classes, data sources
   - **Section 2 — Agent decisions**: what each agent did and why
   - **Section 3 — HITL description**: how many examples flagged, reviewed, corrected
   - **Section 4 — Quality metrics**: EDA summary, quality before/after, annotation agreement, AL curves, final model accuracy/F1
   - **Section 5 — Retrospective**: what worked, what didn't, what would change
3. Create `data/labeled/datacard.md` — data card: format, size, columns, label distribution, provenance
4. Write `run_pipeline.py`: chains all `scripts/` in order via `subprocess.run(["uv", "run", "python", script])`. Include a HITL pause point comment.
5. Verify `pyproject.toml` has all dependencies
