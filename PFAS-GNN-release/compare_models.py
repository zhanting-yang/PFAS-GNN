import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

test_df = pd.read_csv("test.csv", index_col=0)
a = pd.read_csv("large_on_pfas.csv", index_col=0)   # large通用模型
b = pd.read_csv("test_preds.csv", index_col=0)      # 微调模型

rows = []
for t in test_df.columns:
    y = test_df[t].values
    mask = (~pd.isna(y)) & (~pd.isna(a[t].values)) & (~pd.isna(b[t].values))
    y2 = y[mask]
    if len(y2) > 10 and len(np.unique(y2)) == 2:
        auc_a = roc_auc_score(y2, a.loc[mask, t])
        auc_b = roc_auc_score(y2, b.loc[mask, t])
        corr = np.corrcoef(a.loc[mask, t].astype(float), b.loc[mask, t].astype(float))[0, 1]
        rows.append({"任务": t, "AUC_large": round(auc_a,3),
                     "AUC_微调": round(auc_b,3), "预测相关性": round(corr,3)})

rep = pd.DataFrame(rows)
print(rep.to_string(index=False))
print("\n平均 AUC_large =", round(rep["AUC_large"].mean(), 3))
print("平均 AUC_微调  =", round(rep["AUC_微调"].mean(), 3))
print("平均预测相关性  =", round(rep["预测相关性"].mean(), 3))