#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 路线 B · 阶段 1+2：数据装配 + 主分析（先兆 vs 特质决策门）

主分析：线性混合模型检验基线残差是否预测未来 mJSW 丢失速率。
  mJSW_it = (β1 + β2·residual + β3·baseKL + β4·baseJSW)·year + 协变量 + (1+year | knee)
核心系数 β2（residual×year）：
  β2 < 0 且显著 → 先兆（残差预测更快结构进展）
  β2 ≈ 0       → 特质（残差脱钩于结构进展）
"""
from __future__ import annotations
import os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "c2b_data/outputs_to_send_back")
OUT = os.path.join(HERE, "stage2_results")
os.makedirs(OUT, exist_ok=True)

VISIT_MONTH = {"V00": 0, "V01": 12, "V03": 24, "V05": 36, "V06": 48, "V08": 72, "V10": 96}
RESID_SPECS = ["residual_z", "resid_kl_only", "resid_ols_interaction", "resid_quantile_median", "resid_random_forest"]


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std()


def build_long():
    """宽→长 + 合并基线残差/结构/协变量。"""
    base = pd.read_csv(os.path.join(HERE, "c2_derived.csv"))
    jsw = pd.read_csv(os.path.join(D, "C2B_longitudinal_jsw.csv"))

    # 宽→长
    rows = []
    for vv, mo in VISIT_MONTH.items():
        col = f"mjsw_{vv}"
        if col in jsw.columns:
            t = jsw[["ID", "SIDE", col]].rename(columns={col: "mjsw"})
            t["month"] = mo
            rows.append(t)
    long = pd.concat(rows, ignore_index=True)
    long["mjsw"] = pd.to_numeric(long["mjsw"], errors="coerce")
    long = long.dropna(subset=["mjsw"])
    long["year"] = long["month"] / 12.0
    long["knee_id"] = long["ID"].astype(str) + "_" + long["SIDE"].astype(str)

    # 基线键：side 已是 L/R
    bcols = ["ID", "side", "age", "female", "bmi", "cesd", "comorbidity", "nsaid_use",
             "kl_grade", "mjsw", "site"] + [c for c in RESID_SPECS if c in base.columns]
    b = base[bcols].rename(columns={"side": "SIDE", "mjsw": "base_mjsw", "kl_grade": "base_kl"})
    long = long.merge(b, on=["ID", "SIDE"], how="inner")

    # 标准化基线结构（残差已是 z）
    long["base_kl_z"] = zscore(long["base_kl"])
    long["base_mjsw_z"] = zscore(long["base_mjsw"])
    for c in ["age", "bmi", "cesd"]:
        long[c] = pd.to_numeric(long[c], errors="coerce")
    long["female"] = pd.to_numeric(long["female"], errors="coerce")
    return long


def fit_primary(long, resid_col="residual_z"):
    """主混合模型，返回 β2、p、判定。"""
    df = long.dropna(subset=["mjsw", "year", resid_col, "base_kl_z", "base_mjsw_z", "age", "female", "bmi"]).copy()
    df["_resid"] = pd.to_numeric(df[resid_col], errors="coerce")
    formula = ("mjsw ~ year + year:_resid + year:base_kl_z + year:base_mjsw_z "
               "+ age + female + bmi")
    md = smf.mixedlm(formula, df, groups=df["knee_id"], re_formula="~year")
    res = md.fit(method="lbfgs", maxiter=300)
    b2 = res.params.get("year:_resid", np.nan)
    se2 = res.bse.get("year:_resid", np.nan)
    p2 = res.pvalues.get("year:_resid", np.nan)
    ci_lo, ci_hi = b2 - 1.96 * se2, b2 + 1.96 * se2
    return res, dict(spec=resid_col, n_knees=df["knee_id"].nunique(), n_obs=len(df),
                     beta2=b2, se=se2, ci_low=ci_lo, ci_high=ci_hi, p=p2)


def main():
    long = build_long()
    print(f"长表：{len(long)} 观测，{long['knee_id'].nunique()} 膝，"
          f"中位随访点/膝={long.groupby('knee_id').size().median():.0f}")
    print(f"平均 mJSW: V00={long[long.month==0]['mjsw'].mean():.2f} → "
          f"V96={long[long.month==96]['mjsw'].mean():.2f}\n")

    # 主分析
    res, primary = fit_primary(long, "residual_z")
    print("=== 主分析：β2 (residual × year) ===")
    print(f"  β2 = {primary['beta2']:.5f}  (95% CI {primary['ci_low']:.5f} – {primary['ci_high']:.5f})")
    print(f"  p = {primary['p']:.4g}")

    # 全模型年化结构丢失率参考
    b1 = res.params.get("year", np.nan)
    print(f"  参考：平均年化 mJSW 变化 β1(year) = {b1:.4f} mm/年")
    print()

    # 敏感性：跨残差规格
    sens = [primary]
    for spec in RESID_SPECS[1:]:
        if spec in long.columns:
            try:
                _, r = fit_primary(long, spec)
                sens.append(r)
            except Exception as e:
                sens.append(dict(spec=spec, beta2=np.nan, p=np.nan, note=str(e)[:40]))
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(os.path.join(OUT, "table2_mixed_primary_and_sensitivity.csv"), index=False)
    print("=== 跨残差规格敏感性 ===")
    print(sens_df[["spec", "beta2", "ci_low", "ci_high", "p"]].to_string(index=False))
    print()

    # 决策门判定
    b2, p2 = primary["beta2"], primary["p"]
    # 临床意义阈值：β2 对应每 1SD 残差额外年化丢失。换算 96 月累积。
    extra_8yr = b2 * 8  # mm over 8 years per 1 SD residual
    if p2 < 0.05 and b2 < 0:
        verdict = "先兆 (HARBINGER)"
        interp = f"残差每升 1SD，8年额外 mJSW 丢失 {abs(extra_8yr):.3f} mm → 残差预测未来结构损伤"
    elif p2 < 0.05 and b2 > 0:
        verdict = "反向（残差越高结构丢失越慢）"
        interp = f"意外方向，需审视；8年差异 {extra_8yr:.3f} mm"
    else:
        verdict = "特质 (TRAIT)"
        interp = f"残差不显著预测结构进展（p={p2:.3g}）→ 缺口脱钩于关节退变，支持中枢/特质解释"

    print("=" * 60)
    print(f"★ 决策门判定：{verdict}")
    print(f"  {interp}")
    print("=" * 60)

    pd.DataFrame([{"verdict": verdict, "beta2": b2, "p": p2,
                   "extra_mjsw_loss_8yr_per_SD": extra_8yr,
                   "interpretation": interp}]).to_csv(
        os.path.join(OUT, "table2_DECISION_GATE.csv"), index=False)


if __name__ == "__main__":
    main()
