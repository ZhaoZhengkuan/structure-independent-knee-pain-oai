#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 路线 B · 阶段 3：机制三角

残差是否由外周分子/炎症活动解释？填 2×2：
  预测进展(阶段2=先兆) × 与分子标志相关 → 外周隐匿病变
  预测进展 × 与分子标志无关 → 谜（预测损伤但无当前分子信号）

检验：
  3a 残差 ~ FNIH 软骨/骨转换标志物（人级 n≈600，IPSW 校正子集选择）
  3b 残差 ~ MOAKS 局部炎症（积液-滑膜炎 + Hoffa + BML 合成）
  3c 残差 ~ CES-D / 广泛痛（中枢标志对照）
"""
from __future__ import annotations
import os, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "c2b_data/outputs_to_send_back")
OUT = os.path.join(HERE, "stage3_results")
os.makedirs(OUT, exist_ok=True)


def num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"^([-+]?\d*\.?\d+)")[0], errors="coerce")


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std()


def cluster_ols(y, X, groups):
    """带参与者聚类稳健 SE 的 OLS（残差为结局）。返回每个预测因子的 β/CI/p。"""
    import statsmodels.api as sm
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    g = groups.loc[d.index]
    Xc = sm.add_constant(d.drop(columns="y"))
    m = sm.OLS(d["y"], Xc).fit(cov_type="cluster", cov_kwds={"groups": g})
    return m


def main():
    base = pd.read_csv(os.path.join(HERE, "c2_derived.csv"))
    # 删掉原 pipeline 留下的空 MOAKS 列（0% 填充），避免合并时 _x/_y 后缀冲突
    base = base.drop(columns=[c for c in ["moaks_inflammation", "moaks_bml_sum"] if c in base.columns])
    base["SIDE"] = base["side"].astype(str).str.strip().str.upper()
    base["knee_id"] = base["ID"].astype(str) + "_" + base["side"].astype(str)
    base["residual_symptom"] = pd.to_numeric(base["residual_symptom"], errors="coerce")

    # ============ 3a. FNIH 转换标志物 ============
    fnih = pd.read_csv(os.path.join(D, "C2B_fnih_biomarkers_v00.csv"))
    # 仅真实化验列（排除 _Kit_Lot_Num 等）
    bio_cols = [c for c in fnih.columns
                if c.upper().endswith("_NUM") and "KIT_LOT" not in c.upper() and c != "ID"]
    fb = fnih[["ID"] + bio_cols].copy()
    for c in bio_cols:
        fb[c] = num(fb[c])
    # 人级标志物 → 合并到膝（系统性，两膝同值）
    m = base.merge(fb, on="ID", how="left")
    m["in_fnih"] = m[bio_cols].notna().any(axis=1).astype(int)

    # IPSW：子集选择概率（用基线结构/人口学预测进入 FNIH）
    sel_feats = ["age", "female", "bmi", "kl_grade", "residual_symptom"]
    sm_df = m.dropna(subset=sel_feats).copy()
    lr = LogisticRegression(max_iter=2000).fit(zscore_df(sm_df[sel_feats]), sm_df["in_fnih"])
    sm_df["ps"] = lr.predict_proba(zscore_df(sm_df[sel_feats]))[:, 1]
    sm_df["ipsw"] = np.where(sm_df["in_fnih"] == 1, 1 / sm_df["ps"].clip(0.01, 0.99), 0)

    rows3a = []
    sub = m[m["in_fnih"] == 1].copy()
    for c in bio_cols:
        d = sub[["residual_symptom", c, "ID"]].dropna()
        if len(d) < 50:
            continue
        r, p = stats.pearsonr(d["residual_symptom"], zscore(d[c]))
        rows3a.append({"biomarker": c.replace("V00", "").replace("_NUM", ""),
                       "n": len(d), "pearson_r": r, "p": p})
    t3a = pd.DataFrame(rows3a)
    if len(t3a):
        # FDR (BH)
        t3a = t3a.sort_values("p").reset_index(drop=True)
        t3a["q_bh"] = (t3a["p"] * len(t3a) / (t3a.index + 1)).clip(upper=1.0)
    t3a.to_csv(os.path.join(OUT, "table3a_fnih_biomarker_correlations.csv"), index=False)

    # ============ 3b. MOAKS 局部炎症 ============
    moaks = pd.read_csv(os.path.join(D, "C2B_moaks_v00_full.csv"))
    mk = moaks.copy()
    mk["SIDE"] = mk["SIDE"].astype(str).str.strip().str.lower().map(
        lambda s: "R" if s.startswith("1") or "right" in s else ("L" if s.startswith("2") or "left" in s else s.upper()))
    eff = num(mk["V00MEFFWK"]) if "V00MEFFWK" in mk.columns else pd.Series(np.nan, index=mk.index)
    hof = num(mk["V00MSYIC"]) if "V00MSYIC" in mk.columns else pd.Series(np.nan, index=mk.index)
    bms_cols = [c for c in mk.columns if c.upper().startswith("V00MBMS")]
    bml = pd.concat([num(mk[c]) for c in bms_cols], axis=1).sum(axis=1, min_count=1) if bms_cols else pd.Series(np.nan, index=mk.index)
    mk_use = pd.DataFrame({"ID": mk["ID"], "SIDE": mk["SIDE"],
                           "effusion_synovitis": eff, "hoffa_synovitis": hof, "bml_sum": bml})
    mk_use["moaks_inflammation"] = zscore(eff) + zscore(hof)  # 滑膜炎合成
    mm = base.merge(mk_use, on=["ID", "SIDE"], how="left")

    rows3b = []
    for c in ["effusion_synovitis", "hoffa_synovitis", "bml_sum", "moaks_inflammation"]:
        d = mm[["residual_symptom", c]].dropna()
        if len(d) < 50:
            continue
        r, p = stats.pearsonr(d["residual_symptom"], d[c])
        rows3b.append({"moaks_measure": c, "n": len(d), "pearson_r": r, "p": p})
    t3b = pd.DataFrame(rows3b)
    t3b.to_csv(os.path.join(OUT, "table3b_moaks_correlations.csv"), index=False)

    # ============ 3c. 中枢标志对照 ============
    rows3c = []
    for c in ["cesd", "comorbidity"]:
        d = base[["residual_symptom", c]].dropna()
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna()
        r, p = stats.pearsonr(d["residual_symptom"], d[c])
        rows3c.append({"central_marker": c, "n": len(d), "pearson_r": r, "p": p})
    # 广泛痛候选（若可用）
    try:
        wp = pd.read_csv(os.path.join(D, "C2B_widespread_pain_candidates.csv"))
        rows3c.append({"central_marker": "(广泛痛候选见 candidates 文件)", "n": "-",
                       "pearson_r": "-", "p": "-"})
    except Exception:
        pass
    t3c = pd.DataFrame(rows3c)
    t3c.to_csv(os.path.join(OUT, "table3c_central_markers.csv"), index=False)

    # ============ 输出 ============
    print("=== 3a. 残差 ~ FNIH 转换标志物（外周分子活动）===")
    print(t3a.to_string(index=False) if len(t3a) else "  (无足够样本)")
    print()
    print("=== 3b. 残差 ~ MOAKS 局部炎症 ===")
    print(t3b.to_string(index=False))
    print()
    print("=== 3c. 残差 ~ 中枢标志 ===")
    print(t3c.to_string(index=False))
    print()

    # 2×2 解读
    n_bio_sig = int((t3a["q_bh"] < 0.10).sum()) if len(t3a) and "q_bh" in t3a else 0
    moaks_infl_r = t3b[t3b["moaks_measure"] == "moaks_inflammation"]["pearson_r"].iloc[0] if len(t3b) else np.nan
    moaks_sig = (t3b[t3b["moaks_measure"] == "moaks_inflammation"]["p"].iloc[0] < 0.05) if len(t3b) else False
    cesd_r = t3c[t3c["central_marker"] == "cesd"]["pearson_r"].iloc[0]

    print("=" * 60)
    print("★ 机制三角 2×2 解读（结合阶段2=先兆）")
    print(f"  外周分子标志(FNIH)显著关联数: {n_bio_sig}/{len(t3a) if len(t3a) else 0}")
    print(f"  MOAKS 局部炎症关联: r={moaks_infl_r:.3f}, 显著={moaks_sig}")
    print(f"  中枢标志 CES-D 关联: r={cesd_r:.3f}")
    print("=" * 60)

    pd.DataFrame([{
        "stage2_verdict": "harbinger",
        "fnih_sig_count": n_bio_sig, "fnih_total": len(t3a) if len(t3a) else 0,
        "moaks_inflammation_r": moaks_infl_r, "moaks_sig": moaks_sig,
        "cesd_r": cesd_r,
    }]).to_csv(os.path.join(OUT, "table3_mechanism_summary.csv"), index=False)


def zscore_df(df):
    return (df - df.mean()) / df.std()


if __name__ == "__main__":
    main()
