# -*- coding: utf-8 -*-
"""
PFAS 26-assay 多任务 GNN 训练脚本（Chemprop）
用法: py -3.12 train_pfas.py small   ← 先跑这个（PFAS专属模型）
      py -3.12 train_pfas.py large   ← 之后跑（通用对照模型，建议放Colab GPU）
"""
CHEMPROP = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\Scripts\chemprop.exe"
import sys, os, subprocess, glob, random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

# ========== 只需要改这里：填你的文件名和sheet名 ==========
CONFIG = {
    "small": {"file": "si_small.xlsx", "sheet": "pcba_c3f6", "out": "PFAS_small"},
    "large": {"file": "si_large.xlsx", "sheet": "pcba_cf",   "out": "PFAS_large_clean"},
}
# =======================================================

mode = sys.argv[1] if len(sys.argv) > 1 else "small"
cfg = CONFIG[mode]
print(f"模式：{mode} | 文件：{cfg['file']} | sheet：{cfg['sheet']}")

# ---------- 1. 读取 ----------
df = pd.read_excel(cfg["file"], sheet_name=cfg["sheet"])
smiles_col = [c for c in df.columns if c.lower() == "smiles"][0]
task_cols = [c for c in df.columns if c.startswith("PCBA")]
print(f"原始数据：{len(df)} 行 × {len(task_cols)} 个任务")

# ---------- 2. 清洗：去盐、标准化、去重 ----------
remover = SaltRemover.SaltRemover()
def clean(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None: return None
    mol = remover.StripMol(mol)
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)

df["smiles"] = df[smiles_col].apply(clean)
df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])
df = df.set_index("smiles")
print(f"清洗后：{len(df)} 个唯一分子")
# ---- 去污染：large 训练集剔除 small 数据集中所有分子（防止数据泄漏）----
if mode == "large":
    raw_small = pd.read_excel(CONFIG["small"]["file"],
                              sheet_name=CONFIG["small"]["sheet"])["smiles"]
    small_clean = set(raw_small.astype(str).apply(clean).dropna())
    before = len(df)
    df = df[~df.index.isin(small_clean)]
    print(f"去污染：剔除 small 分子 {before - len(df)} 个，剩余 {len(df)} 个")

# ---------- 3. 防同系物泄漏的划分 ----------
# 环状分子按Murcko骨架分组；无环PFAS（大多数！）按碳链长度分组
# → 同一链长的同系物不会同时出现在训练集和测试集
def group_key(smi):
    mol = Chem.MolFromSmiles(smi)
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaf: return "S_" + scaf
    return f"C_{mol.GetNumHeavyAtoms()}"   # 无环 → 按重原子数（≈链长）分组

SPLIT_MODE = sys.argv[2] if len(sys.argv) > 2 else "scaffold"
FT_CKPT = sys.argv[3] if len(sys.argv) > 3 else ""
if SPLIT_MODE == "random":
    df["grp"] = [f"R_{i}" for i in range(len(df))]   # 每个分子独立成组=随机划分
else:
    df["grp"] = [group_key(s) for s in df.index]
groups = list(df.groupby("grp").indices.keys())
random.seed(42); random.shuffle(groups)
n = len(groups)
g_train = set(groups[:int(0.8*n)]); g_val = set(groups[int(0.8*n):int(0.9*n)])

train_df = df[df["grp"].isin(g_train)].drop(columns=["grp"])
val_df   = df[df["grp"].isin(g_val)].drop(columns=["grp"])
test_df  = df[~df.index.isin(train_df.index) & ~df.index.isin(val_df.index)].drop(columns=["grp"])
print(f"划分：train={len(train_df)}, val={len(val_df)}, test={len(test_df)}（组数={n}）")

train_df.to_csv("train.csv"); val_df.to_csv("val.csv"); test_df.to_csv("test.csv")

# ---------- 4. Chemprop 多任务训练 ----------
tasks = " ".join(task_cols)
cmd = (f'"{CHEMPROP}" train --data-path train.csv '
       f'--smiles-column smiles --target-columns {tasks} '
       f'--task-type classification '
       f'{"--checkpoint " + FT_CKPT + " " if FT_CKPT else ""}'
       f'--split-sizes 0.9 0.1 0.0 '
       f'--epochs 100 --patience 10 --output-dir ./{cfg["out"]}_{SPLIT_MODE}')
print("\n开始训练（小数据集CPU约20-40分钟）...\n")
subprocess.run(cmd, shell=True)

# ---------- 5. 评估：逐任务 AUC / AUPRC ----------
ckpt_all = glob.glob(f'./{cfg["out"]}_{SPLIT_MODE}/**/*.pt', recursive=True)
best = [c for c in ckpt_all if 'best' in os.path.basename(c)]
ckpt = (best or ckpt_all)
if not ckpt:
    print("未找到模型文件，训练可能失败，把上方报错发给我"); sys.exit(1)
ckpt = ckpt[0]
print("模型：", ckpt)

pd.DataFrame({"smiles": test_df.index}).to_csv("test_smiles.csv", index=False)
subprocess.run(f'"{CHEMPROP}" predict --test-path test_smiles.csv '
               f'--model-path {ckpt} --preds-path test_preds.csv', shell=True)

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
preds = pd.read_csv("test_preds.csv", index_col=0)
rows = []
for t in task_cols:
    y = test_df[t].values
    mask = (~pd.isna(y)) & (~pd.isna(preds[t].values)) if t in preds.columns else (~pd.isna(y))
    y = y[mask]
    if len(y) > 10 and len(np.unique(y)) == 2:
        rows.append({"任务": t, "n": len(y), "阳性率": round(y.mean(),3),
                     "AUC": round(roc_auc_score(y, preds.loc[mask, t]), 3),
                     "AUPRC": round(average_precision_score(y, preds.loc[mask, t]), 3)})
report = pd.DataFrame(rows).sort_values("AUC", ascending=False)
print("\n===== 测试集逐任务评估 =====")
print(report.to_string(index=False))
print(f"\n平均 AUC = {report['AUC'].mean():.3f}  平均 AUPRC = {report['AUPRC'].mean():.3f}")
print("（对照：Cheng & Ng 报道最好模型平均 AUC ≈ 0.916）")
# ---- 训练集评估（诊断用）----
pd.DataFrame({"smiles": train_df.index}).to_csv("train_smiles.csv", index=False)
subprocess.run(f'"{CHEMPROP}" predict --test-path train_smiles.csv '
               f'--model-path {ckpt} --preds-path train_preds.csv', shell=True)
tr_preds = pd.read_csv("train_preds.csv", index_col=0)
tr_aucs = []
for t in task_cols:
    y = train_df[t].values
    mask = (~pd.isna(y)) & (~pd.isna(tr_preds[t].values))
    y = y[mask]
    if len(y) > 10 and len(np.unique(y)) == 2:
        tr_aucs.append(roc_auc_score(y, tr_preds.loc[mask, t]))
print(f"\n>>> 诊断：训练集平均AUC = {sum(tr_aucs)/len(tr_aucs):.3f}")
report.to_csv(f"report_{mode}_{SPLIT_MODE}.csv", index=False)
print(f"\n完成！模型文件: {ckpt}")
print("注册为ADMET-AI子模型：把该文件拷入 models_dir/PFAS_26assay/ 文件夹")