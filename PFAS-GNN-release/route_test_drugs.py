import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def is_pfas(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    n_f = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F")
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0 or n_f == 0:
        return False
    fluoro_frac = n_f / n_heavy
    perfluoroaryl = False
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(a.GetSymbol() == "C" for a in ring_atoms) and len(ring_atoms) >= 5:
            f_cnt = sum(1 for a in ring_atoms
                        if any(nb.GetSymbol() == "F" for nb in a.GetNeighbors()))
            if f_cnt >= len(ring_atoms) - 1:
                perfluoroaryl = True
                break
    n_polyfluoro_c = sum(
        1 for a in mol.GetAtoms()
        if a.GetSymbol() == "C"
        and sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "F") >= 2)
    if perfluoroaryl or n_polyfluoro_c >= 3:
        return True
    if n_polyfluoro_c >= 1 and fluoro_frac >= 0.15:
        return True
    return False

PATH = r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\Lib\site-packages\admet_ai\resources\data\drugbank_approved.csv"
db = pd.read_csv(PATH)

smi_col = [c for c in db.columns if c.lower() in ("smiles", "canonical_smiles", "structure")][0]
name_candidates = [c for c in db.columns if c.lower() in ("name", "drug_name", "generic_name", "compound_name")]
name_col = name_candidates[0] if name_candidates else None

db = db.dropna(subset=[smi_col]).drop_duplicates(subset=[smi_col]).reset_index(drop=True)
db["fluorinated"] = db[smi_col].astype(str).str.contains("F")

fluoro = db[db["fluorinated"]]
n_fluoro = len(fluoro)
n_other = max(0, 100 - n_fluoro)
n_nonf = int((~db["fluorinated"]).sum())
other = db[~db["fluorinated"]].sample(n=min(n_other, n_nonf), random_state=42)
test_set = pd.concat([fluoro, other]).reset_index(drop=True)
print(f"药物库总数：{len(db)}，其中含氟药物：{n_fluoro}")
print(f"测试集：{len(test_set)} 个（含氟 {n_fluoro} + 非含氟 {len(other)}）")
test_set.to_csv(r"C:\Users\HUAWEI\Desktop\pfas\drugs_test.csv", index=False)

fp = []
for _, row in test_set.iterrows():
    smi = str(row[smi_col])
    if is_pfas(smi):
        nm = str(row[name_col]) if name_col else "(no name)"
        fp.append((nm, smi))

print(f"\n被误判为 PFAS 的药物（假阳性）：{len(fp)} 个")
for nm, smi in fp:
    print(f"  {nm}: {smi}")
print(f"\n特异度：{(len(test_set)-len(fp))/len(test_set)*100:.1f}%  ({len(test_set)-len(fp)}/{len(test_set)})")