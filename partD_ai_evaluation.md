
# AI-Augmented Task

## Prompt
Write a NumPy function that performs IQR-based outlier detection on each column of a 2D array, replacing outliers with the column median.

## AI Output
```python
import numpy as np

def replace_outliers_iqr(data):
    q1 = np.percentile(data,25,axis=0)
    q3 = np.percentile(data,75,axis=0)
    iqr = q3-q1

    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr

    median = np.median(data,axis=0)

    mask = (data < lower) | (data > upper)

    result = data.copy()
    result[mask] = median

    return result
```

## Evaluation
The function correctly computes Q1 and Q3 along axis=0 which ensures the IQR is calculated per column. The lower and upper bounds are also computed correctly using the standard 1.5×IQR rule.

However, the replacement step is partially incorrect. Since `median` is a 1D array of column medians, assigning `result[mask] = median` may misalign values because NumPy will flatten the mask and broadcast incorrectly.

The implementation is vectorized since it avoids explicit Python loops and relies on NumPy broadcasting. A safer approach would replace values column‑wise using advanced indexing or `np.where`.

Testing on synthetic data with injected extreme values confirms that outliers are detected but replacement may not always map to the correct column median. Improving the assignment logic would make the function robust.
