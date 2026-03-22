# Quality Comparison: Raw vs Cleaned

## Strategy Applied

**Conservative**: median fill for missing, drop duplicates, clip IQR outliers

## Metrics Comparison

| Metric | Raw | Cleaned | Delta |
|--------|-----|---------|-------|
| Rows | 127600 | 127600 | 0 |
| Missing (total) | 0 | 0 | 0 |
| Duplicates | 0 | 19 | +19 |
| Text dupes | 0 | 20 | +20 |
| Mean text len | 236.4 | 233.5 | — |
| Std text len | 66.4 | 55.3 | — |
| Min text len | 100 | 100 | 0 |
| Max text len | 1012 | 371 | -641 |
| IQR outliers | 4493 | 0 | -4493 |
| Z>3 outliers | 1755 | 0 | -1755 |

## Class Distribution

| Class | Raw | Cleaned |
|-------|-----|---------|
| Business | 31900 | 31900 |
| Sci/Tech | 31900 | 31900 |
| Sports | 31900 | 31900 |
| World | 31900 | 31900 |

## Justification

The **conservative strategy** is appropriate for this text classification task because:

1. **No data loss**: All 127,600 rows are preserved — important for maintaining balanced classes.
2. **Text clipping over removal**: Long articles are truncated rather than discarded. For news classification, the key information (topic indicators) typically appears early in the text, so truncation preserves classification signal.
3. **Balanced classes maintained**: The 4-way perfect balance (25% each) is preserved, avoiding the need for resampling or class weighting.
4. **Reduced variance**: Clipping reduces text length std from 66.4 to 55.3, which can improve model training stability.
