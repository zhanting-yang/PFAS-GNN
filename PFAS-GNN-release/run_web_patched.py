# -*- coding: utf-8 -*-
"""ADMET-AI v2 增强启动器：
补丁1: DrugBank percentile 跳过自定义任务（防KeyError）
补丁2: 化学空间感知路由 —— PFAS分子显示7轴ADME-Tox雷达(带Routed标题)，其余显示官方雷达
"""
import admet_ai.admet_model as am
from scipy.stats import percentileofscore
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ============ 补丁1: percentile 修复 ============
def _safe_percentiles(self, preds, smiles):
    import pandas as pd
    if getattr(self, "drugbank", None) is None:
        return preds
    drugbank_suffix = am.get_drugbank_suffix(atc_code=self.atc_code)
    drugbank_percentiles = {}
    for property_name in preds.columns:
        if property_name not in self.drugbank_atc_filtered.columns:
            continue
        drugbank_percentiles[f"{property_name}_{drugbank_suffix}"] = [
            percentileofscore(self.drugbank_atc_filtered[property_name], value)
            for value in preds[property_name].values
        ]
    if not drugbank_percentiles:
        return preds
    return pd.concat([preds, pd.DataFrame(drugbank_percentiles, index=smiles)], axis=1)

am.ADMETModel._add_drugbank_percentiles = _safe_percentiles

# ============ 路由规则 v2 ============
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

CATEGORY = {
    "Metabolism":           ["PCBA-883", "PCBA-884", "PCBA-891", "PCBA-1030"],
    "Gene regulation":      ["PCBA-2546", "PCBA-2551", "PCBA-504332", "PCBA-504444", "PCBA-504467", "PCBA-588855", "PCBA-651635"],
    "Cytotoxicity":         ["PCBA-686970", "PCBA-686978", "PCBA-686979"],
    "Membrane targets":     ["PCBA-1461", "PCBA-624417", "PCBA-938"],
    "Protein interaction":  ["PCBA-1468", "PCBA-1688", "PCBA-504339"],
    "Cell signaling":       ["PCBA-624288", "PCBA-720504"],
    "Pathogen/DNA stress":  ["PCBA-540276", "PCBA-720579", "PCBA-720580", "PCBA-624296"],
}

# ============ 补丁2: 雷达路由 ============
import admet_ai.web.app.views as views_mod
import numpy as np
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_orig_get_smiles = views_mod.get_smiles_from_request
_smiles_queue = []

def _get_smiles():
    global _smiles_queue
    all_smiles, error = _orig_get_smiles()
    # 与 views.index 相同的过滤逻辑，保证队列与雷达绘制顺序一致
    _smiles_queue = [s for s in all_smiles
                     if Chem.MolFromSmiles(s) is not None and " " not in s]
    return all_smiles, error

views_mod.get_smiles_from_request = _get_smiles

def _panel_radar_svg(property_id_to_pred) -> bytes:
    """用26个PCBA预测值生成7轴ADME-Tox雷达SVG"""
    cat_vals, labels = [], []
    for cat, cols in CATEGORY.items():
        vals = [property_id_to_pred[c] for c in cols if c in property_id_to_pred]
        cat_vals.append(float(np.mean(vals)) if vals else 0.0)
        labels.append(cat)
    vals = np.concatenate([cat_vals, [cat_vals[0]]])
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, "o-", lw=2, color="crimson")
    ax.fill(angles, vals, alpha=0.3, color="crimson")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Routed → PFAS_Tox21_Panel\n(PFAS detected, ADME-Tox profile)",
                 fontsize=11, color="crimson", fontweight="bold", pad=22)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

_orig_radial = views_mod.plot_radial_summary

def _routed_radial(property_id_to_percentile=None, percentile_suffix=None, **kw):
    global _smiles_queue
    smiles = _smiles_queue.pop(0) if _smiles_queue else None
    if smiles is not None and is_pfas(smiles):
        return _panel_radar_svg(property_id_to_percentile)
    return _orig_radial(property_id_to_percentile=property_id_to_percentile,
                        percentile_suffix=percentile_suffix, **kw)

views_mod.plot_radial_summary = _routed_radial

# ============ 官方启动入口 ============
from admet_ai.web.run import admet_web
admet_web(host="127.0.0.1", port=5000)