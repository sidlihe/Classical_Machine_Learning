1) Problem framing first

Before any model, define:
Target variable: binary, multiclass, multilabel?
Positive class: what class matters most?
Error cost: is false negative worse than false positive?
Success metric: accuracy, F1, recall, PR-AUC, MCC?
Latency / memory constraints: needed for production choices.
Interpretability requirement: regulated domain or not?
Example:
For fraud detection, false negatives are expensive, so recall and PR-AUC often matter more than raw accuracy.