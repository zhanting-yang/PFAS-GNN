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

    if perfluoroaryl:
        return True
    if n_polyfluoro_c >= 3:
        return True
    if n_polyfluoro_c >= 1 and fluoro_frac >= 0.15:
        return True
    return False


# ===== 先用 3 个已知分子验证脚本里确实是 v2 =====
assert is_pfas("O=C(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F") == True, "PFOA应检出!"
assert is_pfas("FC(F)Oc1ccc(NC(=S)Nc2ccc(OC(F)F)cc2)cc1") == True, "CHF2型应检出!"
assert is_pfas("COCCc1ccc(OCC(O)CNC(C)C)cc1") == False, "美托洛尔应排除!"
print("v2 自检通过（PFOA✓ CHF2型✓ 美托洛尔✓）")

# ===== 正式测测试集 =====
df = pd.read_csv(r"C:\Users\HUAWEI\Desktop\pfas\test.csv", index_col=0)
smiles = list(df.index)
hits = [s for s in smiles if is_pfas(s)]
print(f"测试集 PFAS 总数：{len(smiles)}")
print(f"识别为 PFAS：{len(hits)}")
print(f"检出率（敏感性）：{len(hits)/len(smiles)*100:.1f}%")
if len(hits) < len(smiles):
    print("漏检的分子：")
    for s in smiles:
        if s not in hits:
            print(" ", s)