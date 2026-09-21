import subprocess, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

CHEMPROP = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\Scripts\chemprop.exe"
test_df = pd.read_csv("test.csv", index_col=0)   # 当前=scaffold版PFAS测试集
pd.DataFrame({"smiles": test_df.index}).to_csv("pfas_test_smiles.csv", index=False)
subprocess.run(f'"{CHEMPROP}" predict --test-path pfas_test_smiles.csv '
               f'--model-path PFAS_large_clean_random/model_0/best.pt '
               f'--preds-path large_on_pfas.csv', shell=True)
preds = pd.read_csv("large_on_pfas.csv", index_col=0)
aucs = []
for t in test_df.columns:
    y = test_df[t].values
    mask = (~pd.isna(y)) & (~pd.isna(preds[t].values))
    y = y[mask]
    if len(y) > 10 and len(np.unique(y)) == 2:
        aucs.append(roc_auc_score(y, preds.loc[mask, t]))
print(f"large通用模型在PFAS(scaffold)测试集上平均AUC = {sum(aucs)/len(aucs):.3f}")