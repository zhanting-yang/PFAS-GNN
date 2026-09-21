from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def is_pfas(smiles: str) -> bool:
    """规则v2：多氟碳(允许CHF2) + 全氟芳基 + 占比/数量双阈值"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    n_f = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F")
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0 or n_f == 0:
        return False
    fluoro_frac = n_f / n_heavy

    # ① 全氟芳基：纯碳环上 F 取代数 >= 环原子数-1（如五氟苯基）
    perfluoroaryl = False
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(a.GetSymbol() == "C" for a in ring_atoms) and len(ring_atoms) >= 5:
            f_cnt = sum(1 for a in ring_atoms
                        if any(nb.GetSymbol() == "F" for nb in a.GetNeighbors()))
            if f_cnt >= len(ring_atoms) - 1:
                perfluoroaryl = True
                break

    # ② 多氟碳数量：任何带>=2个F的碳都算（含CHF2，不再排除含氢碳）
    n_polyfluoro_c = sum(
        1 for a in mol.GetAtoms()
        if a.GetSymbol() == "C"
        and sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "F") >= 2)

    if perfluoroaryl:
        return True
    if n_polyfluoro_c >= 3:          # PFOA 型：多个多氟碳
        return True
    if n_polyfluoro_c >= 1 and fluoro_frac >= 0.15:   # CF3/CHF2 + 整体富氟
        return True
    return False


if __name__ == "__main__":
    tests = {
        "PFOA(全氟辛酸)": "O=C(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "PFOS": "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "GenX": "O=C(O)C(F)(F)OC(F)(F)C(F)(F)F",
        "氟调聚醇型(CHF2)": "FC(F)Oc1ccc(NC(=S)Nc2ccc(OC(F)F)cc2)cc1",
        "五氟苯基酰胺": "O=C(Nc1cccc(C(=O)Nc2c(F)c(F)c(F)c(F)c2F)c1)S(=O)(=O)c1ccccc1",
        "美托洛尔(药物)": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "氟西汀(CF3药物)": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
        "没食子酸(中药)": "C1=C(C=C(C(=C1O)O)O)C(=O)O",
    }
    for name, smi in tests.items():
        print(f"{name:22s} → {'PFAS' if is_pfas(smi) else '非PFAS'}")