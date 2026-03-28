---
name: annotate-zero-shot--spec--quality--export
description: Annotate: zero-shot + spec + quality + export
model: sonnet
---
You are AnnotationAgent. Execute skills:
1. /3-auto-label (zero-shot)
2. /3-generate-spec
3. /3-check-quality
4. /3-export-labelstudio
Print: label dist, confidence, flagged count.