# -*- coding: utf-8 -*-
"""经典基线：Morgan指纹 + 随机森林（与GNN对比用）"""
import sys, random
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

mode = sys.argv[1] if len(sys.argv) > 1 else "small"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "scaffold"
FILE = "si_small.xlsx" if mode == "small" else "si_large.xlsx"
SHEET = "pcba_c3f6" if mode == "small" else "pcba_cf"

df = pd.read_excel(FILE, sheet_name=SHEET)
smiles_col = [c for c in df.columns if c.lower() == "smiles"][0]
task_cols = [c for c in df.columns if c.startswith("PCBA")]

remover = SaltRemover.SaltRemover()
def clean(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None: return None
    mol = remover.StripMol(mol)
    mol = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)

def fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

df["smiles"] = df[smiles_col].apply(clean)
df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).set_index("smiles")

if mode == "large":
    raw_small = pd.read_excel("si_small.xlsx", sheet_name="pcba_c3f6")["smiles"]
    small_clean = set(raw_small.astype(str).apply(clean).dropna())
    df = df[~df.index.isin(small_clean)]

# 与GNN完全相同的划分
if SPLIT == "random":
    grp = [f"R_{i}" for i in range(len(df))]
else:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    grp = []
    for s in df.index:
        mol = Chem.MolFromSmiles(s)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        grp.append("S_" + scaf if scaf else f"C_{mol.GetNumHeavyAtoms()}")
df["grp"] = grp
groups = list(df.groupby("grp").indices.keys())
random.seed(42); random.shuffle(groups)
n = len(groups)
g_train = set(groups[:int(0.8*n)]); g_val = set(groups[int(0.8*n):int(0.9*n)])
train_df = df[df["grp"].isin(g_train)].drop(columns=["grp"])
test_df  = df[~df.index.isin(train_df.index) & ~df["grp"].isin(g_val)].drop(columns=["grp"])

X_train = np.array([fp(s) for s in train_df.index])
X_test  = np.array([fp(s) for s in test_df.index])
print(f"训练集 {len(train_df)} / 测试集 {len(test_df)}（{SPLIT} 划分）")

rows = []
for t in task_cols:
    y_train = train_df[t].values
    m_tr = ~pd.isna(y_train)
    y_test = test_df[t].values
    m_te = ~pd.isna(y_test)
    if m_tr.sum() < 20 or m_te.sum() <= 10 or len(np.unique(y_test[m_te])) < 2:
        continue
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=42)
    clf.fit(X_train[m_tr], y_train[m_tr].astype(int))
    prob = clf.predict_proba(X_test[m_te])[:, 1]
    rows.append({"任务": t, "n": int(m_te.sum()),
                 "AUC": round(roc_auc_score(y_test[m_te].astype(int), prob), 3)})
rep = pd.DataFrame(rows)
print(rep.sort_values("AUC", ascending=False).to_string(index=False))
print(f"\n平均 AUC = {rep['AUC'].mean():.3f}")
rep.to_csv(f"report_RF_{mode}_{SPLIT}.csv", index=False)