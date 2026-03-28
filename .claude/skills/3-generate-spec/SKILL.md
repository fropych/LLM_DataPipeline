---
name: 3-generate-spec
description: Generate annotation specification — task description, class definitions, examples, edge cases
allowed-tools: Read, Write
---

Generate annotation specification from `data/labeled/dataset.parquet`.

1. Read the labeled dataset, analyze label distribution and examples
2. Write `annotation_spec.md` containing:
   - **Task description**: what is being annotated and why
   - **Classes**: each class with a clear definition
   - **Examples**: 3+ real examples per class (taken from the actual data)
   - **Edge cases**: ambiguous examples and how to handle them
   - **Instructions**: step-by-step guide for human annotators
3. Format as clean Markdown suitable to hand to a colleague for manual annotation
