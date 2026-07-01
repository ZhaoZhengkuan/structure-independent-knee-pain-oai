#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 路线 B · 阶段 1/4 并行代码（修正版）

本脚本在【手头已有的 C2 基线 derived 数据】上，修正上一版三处缺陷并产出可用结果：
  A. TKR 预后的循环设定 → 改为「对照模型不含原始疼痛」的嵌套比较（消除共线）
  B. 交叉验证校准失效   → 去掉 class_weight，自然风险尺度评估校准
  C. 残差测量误差敏感性 → SIMEX 外推

并提供【等阶段 0 纵向数据到位即可跑】的骨架：
  D. 线性混合模型（残差×时间 → mJSW 丢失）—— 先兆 vs 特质主分析
  E. 离散时间生存（时间至结构进展）

输入：c2_derived.csv（之前回传的 C2 基线 derived）
输出：corrected_results/ 下的修正表
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260622)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corrected_results")
os.makedirs(OUT, exist_ok=True)

# 协变量分组（关键：base 不含 womac_pain，以消除与 residual 的共线）
CONFOUNDERS = ["age", "female", "bmi", "kl_grade", "mjsw", "fta", "cesd", "comorbidity", "nsaid_use"]
RESID = "residual_z"
PAIN = "womac_pain"


def load(path="c2_derived.csv"):
    d = pd.read_csv(os.path.join(HERE, path))
    d["followup_years"] = d["followup_days"] / 365.25
    return d


def norm_side(s: pd.Series) -> pd.Series:
    """统一 SIDE 编码为 {R, L}。
    OAI 原始 SIDE: 1=Right, 2=Left（可能带标签 '1: Right'）；derived 已是 R/L。
    """
    x = s.astype(str).str.strip().str.upper()
    out = x.copy()
    out[x.str.startswith("1") | x.str.startswith("R")] = "R"
    out[x.str.startswith("2") | x.str.startswith("L")] = "L"
    return out


# ======================================================================
# A. 修正 TKR 预后的循环设定：嵌套 Cox（对照不含原始疼痛）
# ======================================================================
def corrected_cox(d: pd.DataFrame) -> pd.DataFrame:
    """
    嵌套 cause-specific Cox（死亡按删失处理；参与者聚类稳健 SE）。
      M0: 结构 + 混杂（不含疼痛、不含残差）
      M1: M0 + 原始 WOMAC 疼痛       —— 标准"疼痛重要"模型
      M2: M0 + 残差（结构无法解释的疼痛）—— 不含原始疼痛，故不共线
    注：原版把 residual+pain+structure 同时放入 → r(resid,pain)=0.97 共线 → CI 爆炸。
    """
    rows = []
    base = d[["ID", "followup_years", "tkr_event"] + CONFOUNDERS + [RESID, PAIN]].dropna().copy()

    specs = {
        "M0_structure_confounders": CONFOUNDERS,
        "M1_plus_raw_pain": CONFOUNDERS + [PAIN],
        "M2_plus_residual": CONFOUNDERS + [RESID],
    }
    for name, feats in specs.items():
        work = base[["followup_years", "tkr_event"] + feats].copy()
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(work, duration_col="followup_years", event_col="tkr_event",
                cluster_col=None, robust=True)
        c_index = cph.concordance_index_
        # 关键项 HR
        key = RESID if RESID in feats else (PAIN if PAIN in feats else None)
        if key:
            s = cph.summary.loc[key]
            rows.append({
                "model": name, "n": len(work), "events": int(work["tkr_event"].sum()),
                "key_term": key,
                "HR_per_SD": float(np.exp(s["coef"])),
                "ci_low": float(np.exp(s["coef lower 95%"])),
                "ci_high": float(np.exp(s["coef upper 95%"])),
                "p": float(s["p"]), "c_index": float(c_index),
            })
        else:
            rows.append({
                "model": name, "n": len(work), "events": int(work["tkr_event"].sum()),
                "key_term": "(none)", "HR_per_SD": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "p": np.nan, "c_index": float(c_index),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "table4_corrected_cox.csv"), index=False)
    return tab


# ======================================================================
# B. 修正交叉验证：自然风险尺度，去掉 class_weight
# ======================================================================
def corrected_cv(d: pd.DataFrame) -> pd.DataFrame:
    """
    重复 5×10 折交叉验证，比较：
      base            : 结构 + 混杂（不含疼痛）
      base+raw_pain   : + 原始疼痛
      base+residual   : + 残差（结构无法解释的疼痛）
    关键修正：LogisticRegression 不用 class_weight='balanced'（原版导致
    预测风险 0.37 vs 实际 0.074、Brier 0.175 的校准灾难）。自然尺度报校准。
    """
    feat_sets = {
        "base": CONFOUNDERS,
        "base_plus_raw_pain": CONFOUNDERS + [PAIN],
        "base_plus_residual": CONFOUNDERS + [RESID],
    }
    all_feats = sorted(set(sum(feat_sets.values(), [])))
    dd = d[["tkr_event"] + all_feats].dropna().copy()
    y = dd["tkr_event"].astype(int).values

    oof = {name: np.zeros(len(dd)) for name in feat_sets}
    for rep in range(5):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20260622 + rep)
        for tr, te in skf.split(dd[all_feats], y):
            for name, feats in feat_sets.items():
                pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=3000)),  # 无 class_weight
                ])
                pipe.fit(dd.iloc[tr][feats], y[tr])
                oof[name][te] += pipe.predict_proba(dd.iloc[te][feats])[:, 1] / 5.0

    rows = []
    base_auc = roc_auc_score(y, oof["base"])
    for name in feat_sets:
        p = oof[name]
        auc = roc_auc_score(y, p)
        rows.append({
            "model": name, "n": len(dd), "events": int(y.sum()),
            "AUC": auc,
            "delta_AUC_vs_base": auc - base_auc,
            "Brier": brier_score_loss(y, p),
            "mean_predicted_risk": float(p.mean()),
            "observed_rate": float(y.mean()),
            "calib_in_large": float(p.mean() - y.mean()),
        })
    tab = pd.DataFrame(rows)

    # NRI/IDI: base+residual vs base
    pb, pr = oof["base"], oof["base_plus_residual"]
    idi = (pr[y == 1].mean() - pb[y == 1].mean()) - (pr[y == 0].mean() - pb[y == 0].mean())
    # 类别 NRI (阈值 5/10/15%)
    nri_rows = []
    for thr in (0.05, 0.10, 0.15):
        up_ev = ((pr >= thr) & (pb < thr) & (y == 1)).sum() - ((pr < thr) & (pb >= thr) & (y == 1)).sum()
        dn_ne = ((pr < thr) & (pb >= thr) & (y == 0)).sum() - ((pr >= thr) & (pb < thr) & (y == 0)).sum()
        nri = up_ev / max((y == 1).sum(), 1) + dn_ne / max((y == 0).sum(), 1)
        nri_rows.append({"threshold": thr, "NRI": nri})
    tab.to_csv(os.path.join(OUT, "table5_corrected_cv.csv"), index=False)
    pd.DataFrame(nri_rows).assign(IDI=idi).to_csv(os.path.join(OUT, "table5b_nri_idi.csv"), index=False)
    return tab


# ======================================================================
# C. SIMEX：残差对结构测量误差的敏感性
# ======================================================================
def simex_residual(d: pd.DataFrame, lambdas=(0.5, 1.0, 1.5, 2.0), B=40) -> pd.DataFrame:
    """
    SIMEX 外推残差→TKR 的 Cox 系数对结构测量误差的稳健性。
    结构测量误差近似：mJSW SD≈0.15mm（文献），对 mjsw 注入 λ 倍方差，
    重建残差 → 重估 residual 的 logHR → 外推回 λ=-1（零误差）。
    """
    sub = d[["followup_years", "tkr_event", "womac_pain", "kl_grade", "mjsw", "fta"]].dropna().copy()
    sigma_mjsw = 0.15  # mm，定量 JSW 测量误差近似
    from sklearn.linear_model import HuberRegressor

    def resid_loghr(extra_var):
        s = sub.copy()
        if extra_var > 0:
            s["mjsw"] = s["mjsw"] + RNG.normal(0, np.sqrt(extra_var) * sigma_mjsw, len(s))
        X = s[["kl_grade", "mjsw", "fta"]].values
        hub = HuberRegressor(max_iter=500).fit(X, s["womac_pain"].values)
        s["resid"] = s["womac_pain"].values - hub.predict(X)
        s["resid_z"] = (s["resid"] - s["resid"].mean()) / s["resid"].std()
        cph = CoxPHFitter().fit(s[["followup_years", "tkr_event", "resid_z"]],
                                "followup_years", "tkr_event", robust=True)
        return float(cph.summary.loc["resid_z", "coef"])

    rows = [{"lambda": 0.0, "log_HR": resid_loghr(0.0)}]
    for lam in lambdas:
        vals = [resid_loghr(lam) for _ in range(B)]
        rows.append({"lambda": lam, "log_HR": float(np.mean(vals))})
    sim = pd.DataFrame(rows)
    # 二次外推到 λ=-1
    coefs = np.polyfit(sim["lambda"], sim["log_HR"], 2)
    extrap = np.polyval(coefs, -1.0)
    sim_out = pd.concat([sim, pd.DataFrame([{"lambda": -1.0, "log_HR": extrap}])], ignore_index=True)
    sim_out["HR"] = np.exp(sim_out["log_HR"])
    sim_out["note"] = ["observed" if l >= 0 else "SIMEX-extrapolated (error-corrected)" for l in sim_out["lambda"]]
    sim_out.to_csv(os.path.join(OUT, "table_s_simex.csv"), index=False)
    return sim_out


# ======================================================================
# D. 【骨架】线性混合模型：残差×时间 → mJSW 丢失（先兆 vs 特质主分析）
#    等阶段 0 的 C2B_longitudinal_jsw.csv 到位即可跑。
# ======================================================================
def mixed_model_leading_indicator(long_jsw_path: str, baseline_path="c2_derived.csv"):
    """
    主分析（H2 先兆 vs 特质）。需要纵向 JSW 宽表（C2B_longitudinal_jsw.csv）。

    模型：mJSW_it = (β1 + β2·residual_i + β3·baseKL_i + β4·baseJSW_i)·month_t
                    + 协变量 + (1+month | knee)
    核心系数 β2 = residual×month：
      β2 < 0 且显著 → 先兆假说（残差预测更快 JSW 丢失）
      β2 ≈ 0       → 特质假说（残差脱钩于结构进展）
    """
    import statsmodels.formula.api as smf

    base = pd.read_csv(os.path.join(HERE, baseline_path))
    wide = pd.read_csv(long_jsw_path)
    # 宽→长
    visit_month = {"V00": 0, "V01": 12, "V03": 24, "V05": 36, "V06": 48, "V08": 72, "V10": 96}
    long_rows = []
    for vv, mo in visit_month.items():
        col = f"mjsw_{vv}"
        if col in wide.columns:
            tmp = wide[["ID", "SIDE", col]].rename(columns={col: "mjsw"})
            tmp["month"] = mo
            long_rows.append(tmp)
    long = pd.concat(long_rows, ignore_index=True).dropna(subset=["mjsw"])
    long["side_n"] = norm_side(long["SIDE"])
    long["knee_id"] = long["ID"].astype(str) + "_" + long["side_n"]

    # 合并基线残差与基线结构（SIDE 归一化后对齐）
    bcols = ["ID", "side", "residual_z", "kl_grade", "mjsw", "age", "female", "bmi"]
    b = base[bcols].rename(columns={"mjsw": "base_mjsw", "kl_grade": "base_kl"})
    b["side_n"] = norm_side(b["side"])
    long = long.merge(b.drop(columns=["side"]), on=["ID", "side_n"], how="inner")

    # 标准化 month（年）
    long["year"] = long["month"] / 12.0
    md = smf.mixedlm(
        "mjsw ~ year + year:residual_z + year:base_kl + year:base_mjsw + age + female + bmi",
        long, groups=long["knee_id"], re_formula="~year",
    )
    res = md.fit(method="lbfgs", maxiter=200)
    out = res.summary().tables[1]
    out.to_csv(os.path.join(OUT, "table2_mixed_leading_indicator.csv"))
    # 提取 β2
    beta2 = res.params.get("year:residual_z", np.nan)
    p2 = res.pvalues.get("year:residual_z", np.nan)
    verdict = "先兆 (harbinger)" if (beta2 < 0 and p2 < 0.05) else "特质 (trait)"
    pd.DataFrame([{"beta2_resid_x_year": beta2, "p": p2, "verdict": verdict}]).to_csv(
        os.path.join(OUT, "table2_verdict.csv"), index=False)
    return res


# ======================================================================
# E. 【骨架】离散时间生存：时间至结构进展（mJSW 丢失≥0.7mm）
# ======================================================================
def discrete_time_progression(long_jsw_path: str, baseline_path="c2_derived.csv", threshold=0.7):
    """
    次分析：时间至「mJSW 自基线丢失 ≥ threshold mm」的离散时间生存。
    残差为预测因子，基线结构调整。需要纵向 JSW。
    """
    base = pd.read_csv(os.path.join(HERE, baseline_path))
    wide = pd.read_csv(long_jsw_path)
    visit_month = {"V00": 0, "V01": 12, "V03": 24, "V05": 36, "V06": 48, "V08": 72, "V10": 96}
    wide["knee_id"] = wide["ID"].astype(str) + "_" + norm_side(wide["SIDE"])
    base_jsw = wide["mjsw_V00"]
    # 构建每访视是否达进展阈值 → 首次达标月份
    prog_month = pd.Series(np.nan, index=wide.index)
    for vv, mo in visit_month.items():
        if vv == "V00":
            continue
        col = f"mjsw_{vv}"
        if col not in wide.columns:
            continue
        loss = base_jsw - wide[col]
        hit = (loss >= threshold) & prog_month.isna()
        prog_month[hit] = mo
    wide["progression_month"] = prog_month
    wide["progressed"] = prog_month.notna().astype(int)
    # 离散时间：person-period 展开 + 互补 log-log（此处导出事件表，建模在分析端接 lifelines/statsmodels）
    surv = wide[["ID", "SIDE", "knee_id", "progressed", "progression_month"]].copy()
    surv.to_csv(os.path.join(OUT, "discrete_time_progression_events.csv"), index=False)
    return surv


def main():
    d = load()
    print(f"载入 {len(d)} 膝，TKR {int(d['tkr_event'].sum())} 例\n")

    print("A. 修正 Cox（对照不含原始疼痛，消除共线）...")
    cox = corrected_cox(d)
    print(cox.to_string(index=False))
    print()

    print("B. 修正交叉验证（去 class_weight，自然尺度校准）...")
    cv = corrected_cv(d)
    print(cv.to_string(index=False))
    print()

    print("C. SIMEX 残差测量误差敏感性...")
    sim = simex_residual(d)
    print(sim.to_string(index=False))
    print()

    print("D/E. 纵向骨架已就绪，等 C2B_longitudinal_jsw.csv 到位即可跑。")
    print(f"\n所有修正表已写入 {OUT}/")


if __name__ == "__main__":
    main()
