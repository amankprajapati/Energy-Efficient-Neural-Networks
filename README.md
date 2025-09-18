# Energy-Efficient-Neural-Networks

Experiments to make neural nets **smaller and cheaper** to run with **minimal accuracy loss** via pruning + short fine-tuning.

## Methods included (current notebook)
Exact PyTorch pruning APIs used:
- 
`torch.nn.utils.prune.l1_unstructured` (magnitude/L1 weight pruning)- `torch.nn.utils.prune.global_unstructured` (global magnitude pruning across layers)- `torch.nn.utils.prune.remove` (make masks permanent)

Interpretation:
- These are **fine‑grained (unstructured)** methods (magnitude-based). They zero individual weights and can be made permanent with `prune.remove`.
- Structured/channel or pattern (N:M) pruning is **not** used in this notebook.

## Overall results (summarized from CSVs)
**FCN/MLP**

**CNN**

> CSVs: `mlp_pruning_results.csv`, `cnn_pruning_results.csv` (contain full per‑ratio metrics).

## Run
Open any notebook in Colab or run locally (Python 3.10+, `torch`, `torchvision`). Notebooks may export CSVs and plots automatically.

---
Maintainer: @amankprajapati · License: MIT/Apache-2.0 (add one)
