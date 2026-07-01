#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 路线 B · 单次提取脚本：一次取齐纵向结构 + 机制层变量。

路线 B 的主结局是「未来结构进展」，需要纵向 JSW/KL；机制三角需要 FNIH 生标 + MOAKS。
本脚本一次性提取，跑完打包成单个 zip 回传。

运行：python extract_longitudinal.py
输出（落到 outputs_to_send_back/）：
  - C2B_longitudinal_jsw.csv        ID + SIDE + 各访视 mJSW（宽表）
  - C2B_longitudinal_kl.csv         ID + SIDE + 各访视 KL（宽表）
  - C2B_moaks_v00_full.csv          ID + SIDE + MOAKS V00 全部列（我自己算炎症合成）
  - C2B_fnih_biomarkers_v00.csv     ID + 全部 V00*_NUM 生标
  - C2B_widespread_pain_candidates.csv  AllClinical00 非膝疼痛候选列（可选）
  - C2B_extraction_report.txt       可得性汇总
"""

import os
import re
import zipfile
import pandas as pd

# ============ 只改这一行：你的 OAI 原始 zip 路径 ============
OAI_ZIP = os.environ.get("OAI_ZIP", "/Users/OAI数据库/OAICompleteData_ASCII.zip")
# ==========================================================

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_to_send_back")
os.makedirs(OUT_DIR, exist_ok=True)

VISITS = ["00", "01", "03", "05", "06", "08", "10"]
KL_VISITS = ["00", "01", "03", "05", "06", "08", "10", "12"]
report = []


def log(msg):
    print(msg)
    report.append(msg)


def find_member(z, needle):
    """按 basename 模糊匹配 zip 成员（不区分大小写）。"""
    cands = [
        n for n in z.namelist()
        if n.lower().endswith(".txt") and needle.lower() in os.path.basename(n).lower()
    ]
    return cands[0] if cands else None


def read_table(z, member, usecols=None):
    with z.open(member) as fh:
        return pd.read_csv(
            fh, sep="|", dtype=str, encoding="latin1", low_memory=False, usecols=usecols
        )


def colmap(cols):
    return {c.upper(): c for c in cols}


def normalize_side(x):
    """统一 OAI SIDE 编码，避免 '1' 与 '1: Right' 在宽表合并时重复膨胀。"""
    if pd.isna(x):
        return x
    s = str(x).strip().lower()
    if s.startswith("1") or "right" in s:
        return "R"
    if s.startswith("2") or "left" in s:
        return "L"
    return str(x).strip()


def is_missing_marker(x):
    s = str(x).strip()
    return s == "" or s.lower() == "nan" or s.startswith(".")


def extract_longitudinal_jsw(z):
    """纵向 mJSW：kxr_qjsw_duryea{vv} → V{vv}MCMJSW。"""
    log("\n=== 1. 纵向 mJSW ===")
    merged = None
    for vv in VISITS:
        member = find_member(z, f"kxr_qjsw_duryea{vv}")
        if member is None:
            log(f"  V{vv}: 未找到 kxr_qjsw_duryea{vv}.txt")
            continue
        df = read_table(z, member)
        cm = colmap(df.columns)
        idc = cm.get("ID")
        sidec = cm.get("SIDE")
        jswc = cm.get(f"V{vv}MCMJSW")
        if not (idc and jswc):
            log(f"  V{vv}: 缺 ID 或 V{vv}MCMJSW（该表列：{list(df.columns)[:8]}...）")
            continue
        keep_cols = [idc] + ([sidec] if sidec else []) + [jswc]
        sub = df[keep_cols].copy()
        rename = {idc: "ID", jswc: f"mjsw_V{vv}"}
        if sidec:
            rename[sidec] = "SIDE"
        sub = sub.rename(columns=rename)
        if "SIDE" in sub.columns:
            sub["SIDE"] = sub["SIDE"].map(normalize_side)
        sub = sub[~sub[f"mjsw_V{vv}"].map(is_missing_marker)].copy()
        nn = sub[f"mjsw_V{vv}"].notna().sum()
        log(f"  V{vv}: {nn} 行有 mJSW")
        keys = ["ID", "SIDE"] if "SIDE" in sub.columns else ["ID"]
        sub = sub.drop_duplicates(keys, keep="first")
        merged = sub if merged is None else merged.merge(sub, on=keys, how="outer")
    if merged is not None:
        path = os.path.join(OUT_DIR, "C2B_longitudinal_jsw.csv")
        merged.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"  → 写出 {path}（{len(merged)} 行）")
    else:
        log("  !! 纵向 mJSW 提取失败")


def extract_longitudinal_kl(z):
    """纵向 KL：KXR_SQ_BU{vv} → V{vv}XRKL。
    注意：SQ_BU 表按 ID+SIDE+READPRJ 键控（一膝多读片项目，~16k 行）。
    去重策略：优先主纵向读片项目 READPRJ=15，其次取每膝首个非缺失 KL。
    """
    log("\n=== 2. 纵向 KL（含 READPRJ 去重）===")
    merged = None
    for vv in KL_VISITS:
        member = find_member(z, f"kxr_sq_bu{vv}")
        if member is None:
            member = find_member(z, f"sq_bu{vv}") or find_member(z, f"xrsq{vv}")
        if member is None:
            log(f"  V{vv}: 未找到 KXR_SQ_BU{vv}.txt")
            continue
        df = read_table(z, member)
        cm = colmap(df.columns)
        idc = cm.get("ID")
        sidec = cm.get("SIDE")
        rpc = cm.get("READPRJ")
        klc = cm.get(f"V{vv}XRKL")
        if not (idc and klc):
            log(f"  V{vv}: 缺 ID 或 V{vv}XRKL（列样本：{list(df.columns)[:10]}）")
            continue

        keep_cols = [idc] + ([sidec] if sidec else []) + ([rpc] if rpc else []) + [klc]
        sub = df[keep_cols].copy()
        rename = {idc: "ID", klc: f"kl_V{vv}"}
        if sidec:
            rename[sidec] = "SIDE"
        if rpc:
            rename[rpc] = "READPRJ"
        sub = sub.rename(columns=rename)
        if "SIDE" in sub.columns:
            sub["SIDE"] = sub["SIDE"].map(normalize_side)

        keys = ["ID", "SIDE"] if "SIDE" in sub.columns else ["ID"]
        n_before = len(sub)
        # 去除 KL 缺失行
        kl_blank = sub[f"kl_V{vv}"].map(is_missing_marker) | sub[f"kl_V{vv}"].isna()
        sub_valid = sub[~kl_blank].copy()
        if "READPRJ" in sub_valid.columns:
            sub_valid["_rp"] = pd.to_numeric(
                sub_valid["READPRJ"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
            sub_valid["_pref"] = (sub_valid["_rp"] != 15).astype(int)  # READPRJ=15 优先(_pref=0)
            sub_valid = sub_valid.sort_values(keys + ["_pref"]).drop_duplicates(keys, keep="first")
        else:
            sub_valid = sub_valid.drop_duplicates(keys, keep="first")
        sub_out = sub_valid[keys + [f"kl_V{vv}"]].copy()

        nn = sub_out[f"kl_V{vv}"].notna().sum()
        log(f"  V{vv}: 原 {n_before} 行 → 去重后 {len(sub_out)} 膝，{nn} 有 KL")
        merged = sub_out if merged is None else merged.merge(sub_out, on=keys, how="outer")

    if merged is not None:
        path = os.path.join(OUT_DIR, "C2B_longitudinal_kl.csv")
        merged.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"  → 写出 {path}（{len(merged)} 膝）")
    else:
        log("  !! 纵向 KL 提取失败")


def extract_moaks_v00(z):
    """MOAKS V00 全表（小表，整张导出，炎症合成在分析端算）。"""
    log("\n=== 3. MOAKS V00 全表 ===")
    member = find_member(z, "MOAKS_BICL00")
    if member is None:
        member = find_member(z, "moaks") and find_member(z, "moaks00")
    if member is None:
        log("  未找到 MOAKS_BICL00.txt")
        return
    m = read_table(z, member)
    path = os.path.join(OUT_DIR, "C2B_moaks_v00_full.csv")
    m.to_csv(path, index=False, encoding="utf-8-sig")
    # 报告关键炎症列可得性
    cm = colmap(m.columns)
    for key in ["V00MEFFWK", "V00MSYIC"]:
        c = cm.get(key)
        nn = m[c].notna().sum() if c else 0
        log(f"  {key}: {'有，非缺失 '+str(nn) if c else '缺'}")
    bms = [c for c in m.columns if re.match(r"V00MBMS", c, re.I)]
    log(f"  BML 亚区列（V00MBMS*）: {len(bms)} 个")
    log(f"  → 写出 {path}（{len(m)} 行，{len(m.columns)} 列）")


def extract_fnih_biomarkers(z):
    """FNIH 生标 V00：Biospec_FNIH_Labcorp00 → 全部 V00*_NUM。"""
    log("\n=== 4. FNIH 生标 V00 ===")
    member = find_member(z, "Biospec_FNIH_Labcorp00")
    if member is None:
        log("  未找到 Biospec_FNIH_Labcorp00.txt")
        return
    df = read_table(z, member)
    cm = colmap(df.columns)
    idc = cm.get("ID")
    num_cols = [c for c in df.columns if c.upper().endswith("_NUM") and c.upper().startswith("V00")]
    keep = [idc] + num_cols
    sub = df[keep].rename(columns={idc: "ID"})
    path = os.path.join(OUT_DIR, "C2B_fnih_biomarkers_v00.csv")
    sub.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"  生标 *_NUM 列: {len(num_cols)} 个；样本：{num_cols[:6]}")
    log(f"  → 写出 {path}（{len(sub)} 行）")


def extract_widespread_pain(z):
    """广泛/非膝疼痛候选列（AllClinical00），可选。仅导出候选列名 + 样本供挑选。"""
    log("\n=== 5. 广泛痛候选（可选）===")
    member = find_member(z, "AllClinical00")
    if member is None:
        log("  未找到 AllClinical00.txt")
        return
    with z.open(member) as fh:
        hdr = fh.readline().decode("latin1", errors="replace").rstrip("\r\n").split("|")
    # 非膝部位疼痛相关词；排除明显的膝/WOMAC
    pat = re.compile(r"PAIN|ACHE|HIP|HAND|BACK|FOOT|FEET|SHLD|SHOULDER|NECK|ELBOW|WRIST|ANKLE", re.I)
    excl = re.compile(r"WOM|KP|KNEE|KOOS", re.I)
    cand = [c for c in hdr if pat.search(c) and not excl.search(c)]
    rows = []
    if cand:
        idc = "ID" if "ID" in hdr else hdr[0]
        df = read_table(z, member, usecols=[idc] + cand)
        for c in cand:
            nn = df[c].notna().sum()
            samp = df[c].dropna().astype(str).head(3).tolist()
            rows.append({"column": c, "non_missing": nn, "sample": " | ".join(samp)})
    path = os.path.join(OUT_DIR, "C2B_widespread_pain_candidates.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    log(f"  候选非膝疼痛列: {len(cand)} 个 → 写出 {path}")


def main():
    if not os.path.exists(OAI_ZIP):
        raise SystemExit(f"找不到 OAI zip：{OAI_ZIP}\n请修改脚本顶部的 OAI_ZIP 路径。")
    log(f"打开：{OAI_ZIP}")
    with zipfile.ZipFile(OAI_ZIP) as z:
        extract_longitudinal_jsw(z)
        extract_longitudinal_kl(z)
        extract_moaks_v00(z)
        extract_fnih_biomarkers(z)
        extract_widespread_pain(z)

    rep_path = os.path.join(OUT_DIR, "C2B_extraction_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log(f"\n报告写出：{rep_path}")

    # 自动打包
    zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "C2B_data_to_send_back.zip")
    files = [os.path.join(OUT_DIR, n) for n in os.listdir(OUT_DIR) if not n.startswith(".")]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            zf.write(fp, os.path.join("outputs_to_send_back", os.path.basename(fp)))
    print(f"\n已打包 {len(files)} 个产物 → {zip_path}")
    print("把 C2B_data_to_send_back.zip 整个回传即可。")


if __name__ == "__main__":
    main()
