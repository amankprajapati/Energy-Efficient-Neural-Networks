# MNIST Model Pruning & Fine‑Tuning (FCN & CNN)

**Repo notebook:** `mnist_pruning_22bec006.ipynb`

## What this notebook does

We trained **two models on MNIST** — a **fully‑connected network (FCN/MLP)** and a **convolutional neural network (CNN)** — then **pruned** them at multiple sparsity levels and **fine‑tuned** to recover accuracy. The goal was to show, empirically, that **you don’t need all weights/filters** in a neural network: you can **shrink the model** (fewer active parameters, smaller file size) **while maintaining nearly the same accuracy**.

**Dataset:** MNIST  
**Models:** FCN/MLP and CNN

## Pruning techniques used
- **Fine‑grained (unstructured):** applied to individual weights (e.g., `prune.l1_unstructured`, `global_unstructured`). **Detected:** Yes.
- **Channel/Filter‑level (structured):** removes entire channels/filters (e.g., `prune.ln_structured`). **Detected:** No.
- **Pattern‑based (N:M / block):** enforces fixed patterns within tensors (e.g., 2:4). **Detected:** No.

## Workflow

1. **Baseline training** of FCN and CNN on MNIST to establish reference accuracy and model sizes.  
2. **Apply pruning** at multiple **pruning ratios** (sparsity levels).  
3. **Fine‑tune** each pruned checkpoint for several epochs to recover any lost accuracy.  
4. **Evaluate** on the test set and **log metrics** (accuracy, params, size).  
5. **Export results** to CSVs:  
   - `mlp_pruning_results.csv` for FCN/MLP  
   - `cnn_pruning_results.csv` for CNN

## Results summary (from CSVs)
### FCN / MLP
Top rows below (see CSV for full detail):

| model   |   sparsity |
|:--------|-----------:|
| MLP     |       0.3  |
| MLP     |       0.5  |
| MLP     |       0.7  |
| MLP     |       0.9  |
| MLP     |       0.99 |

### CNN
Top rows below (see CSV for full detail):

| model   |   sparsity |
|:--------|-----------:|
| CNN     |       0.3  |
| CNN     |       0.5  |
| CNN     |       0.7  |
| CNN     |       0.9  |
| CNN     |       0.99 |

## Key findings
- **Baseline test accuracy (MLP):** See table/CSV
- **Baseline test accuracy (CNN):** See table/CSV
- **Accuracy vs sparsity:** moderate pruning ratios preserved accuracy after fine‑tuning; high ratios eventually degrade.
- **Model size/params:** decreased with higher pruning, demonstrating tangible compression benefits.

## How to reproduce

1. Open `mnist_pruning_22bec006.ipynb` in Google Colab.  
2. Run all cells in order: baseline → pruning at chosen ratios → fine‑tuning → evaluation → CSV export.  
3. Adjust `PRUNE_RATIOS` and pruning method calls (e.g., `prune.l1_unstructured`, `prune.ln_structured`) to explore different trade‑offs.

## Notes

- We **did not** use pattern‑based (N:M) pruning unless explicitly shown in the code. If needed, integrate a pattern‑pruning library or custom masks.  
- Structured channel pruning may require tensor re‑wiring or layer surgery to physically shrink tensors for runtime speedups; PyTorch’s masking alone zeros weights but may not reduce compute unless you re‑materialize pruned layers.
