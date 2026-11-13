"""封存批覆盖质量核定：精确后验推断与最优查证/重制策略（仅用标准库）。

方法概述（详见 REPORT.md）：
  * 三个可查证批号（A0/A1/A2）是全部质检与血缘中的"长程"变量；先对它们的
    8 种赋值联合条件化。给定 A 赋值后，其余 50 个批号（B00..B49）上的全部
    有效质检窗口与成果块血缘窗口都是链上连续区间（长度 <= K=6），用链式
    动态规划精确求和：DP 状态 = 最近 K 位缺陷模式 + 截断损失面积，转移质量
    = 各 B 批先验权重（整数，公共分母最后统一约掉）。不枚举 2^53 个缺陷
    世界、不做独立块近似、不做蒙特卡洛；全部为精确有理数运算。
  * 知识状态 = 三项查证的部分答复（三字符编码，? 未查）。对每个正概率状态
    计算 64 个重制子集的达标概率，再用 Bellman 递推求最小期望总费用；
    平局时先结束后按动作 ID 序列字典序。

入口：
  python3 analysis.py run      从 data/ 重算并写入 result.json 与 REPORT.md
  python3 analysis.py verify   重算本批并核对已有产物、运行原回归、用直接枚举
                               的小例判定器互核，并独立重推结果中的门槛/分支/
                               动作/费用递推；不一致则非零退出
  python3 analysis.py demo     展示现状覆盖风险、最优期望费用与策略分支
"""
import json
import subprocess
import sys
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path

import source

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 工具
def fr(x):
    """有理数 -> 最简 n/d 字符串（分母为正，整数写 n/1）。"""
    x = Fraction(x)
    return f"{x.numerator}/{x.denominator}"


# ---------------------------------------------------------------- 链结构
def build_chain(model):
    """把批号分成被查证的 A 集合与链上 B 序列；校验所有窗口连续。"""
    a_batches = [x["batch"] for x in model["inspections"]]  # 顺序 = 查证文件顺序
    a_set = set(a_batches)
    b_order = sorted(b for b in model["priors"] if b not in a_set)
    b_index = {b: i for i, b in enumerate(b_order)}
    pools = []
    for p in model["pools"]:
        bs = sorted(b_index[b] for b in p["batches"] if b in b_index)
        if bs and bs != list(range(bs[0], bs[-1] + 1)):
            raise ValueError("NON_CHAIN_POOL:" + p["id"])
        pools.append({"id": p["id"], "start": bs[0] if bs else 0,
                      "end": bs[-1] if bs else -1, "blen": len(bs),
                      "a": sorted(b for b in p["batches"] if b in a_set),
                      "lo": p["lo"], "hi": p["hi"],
                      "b_names": [b for b in p["batches"] if b in b_index]})
    tiles = []
    for t in model["tiles"]:
        nm = source.names(t["rule"])
        bs = sorted(b_index[b] for b in nm if b in b_index)
        if bs and bs != list(range(bs[0], bs[-1] + 1)):
            raise ValueError("NON_CHAIN_TILE:" + t["id"])
        tiles.append({"id": t["id"], "area": t["area"], "rule": t["rule"],
                      "start": bs[0] if bs else 0, "end": bs[-1] if bs else -1,
                      "blen": len(bs), "a": sorted(b for b in nm if b in a_set),
                      "b_names": [b_order[i] for i in bs]})
    K = max([1] + [p["blen"] for p in pools] + [t["blen"] for t in tiles])
    return {"a_batches": a_batches, "b_order": b_order, "b_index": b_index,
            "pools": pools, "tiles": tiles, "K": K, "n": len(b_order)}


def tile_table(rule, a_names, b_names):
    """成果块规则的局部真值表：下标 ((a_bits,a_rep,b_bits,b_rep)) -> 可用?。"""
    na, nb = len(a_names), len(b_names)
    tab = [False] * (1 << (2 * na + 2 * nb))
    for a_bits in range(1 << na):
        fa = {a_names[i] for i in range(na) if a_bits >> i & 1}
        for a_rep in range(1 << na):
            ra = {a_names[i] for i in range(na) if a_rep >> i & 1}
            base_a = (a_bits << na) | a_rep
            for b_bits in range(1 << nb):
                faults = fa | {b_names[j] for j in range(nb) if b_bits >> j & 1}
                for b_rep in range(1 << nb):
                    repaired = ra | {b_names[j] for j in range(nb) if b_rep >> j & 1}
                    tab[(((base_a << nb) | b_bits) << nb) | b_rep] = source.good(rule, faults, repaired)
    return tab


def build_trans(n, K, pools_at):
    """每个位置、每种当前模式、每种新缺陷位 -> 新模式（-1 表示违反质检）。"""
    trans = []
    for i in range(n):
        pa = pools_at[i]
        row = []
        for pat in range(1 << K):
            base = pat >> 1
            out = []
            for b in (0, 1):
                np_ = base | (b << (K - 1))
                ok = True
                for shift, pmask, lo2, hi2 in pa:
                    c = ((np_ >> shift) & pmask).bit_count()
                    if c < lo2 or c > hi2:
                        ok = False
                        break
                out.append(np_ if ok else -1)
            row.append(tuple(out))
        trans.append(row)
    return trans


def run_dp(n, K, bw, trans, delta_pos, cap, const_delta, snap_ends=()):
    """链式精确求和。

    状态 = (最近 K 位缺陷模式, 截断损失 0..cap)；cap 桶表示“>=cap”。
    转移质量 = B 批先验整数权重 bw[i] = (良好 den-n, 缺陷 n)。
    返回 (总质量, 损失分布[0..cap], 快照{位置: 各模式质量})。
    """
    size = 1 << K
    cur = [None] * size
    row0 = [0] * (cap + 1)
    row0[min(const_delta, cap)] = 1
    cur[0] = row0
    snaps = {}
    for i in range(n):
        nxt = [None] * size
        ti = trans[i]
        dp_i = delta_pos[i]
        w0, w1 = bw[i]
        for pat in range(size):
            row = cur[pat]
            if row is None:
                continue
            for t, wb in ((ti[pat][0], w0), (ti[pat][1], w1)):
                if t < 0:
                    continue
                d = dp_i[t] if dp_i is not None else 0
                nr = nxt[t]
                if nr is None:
                    nr = nxt[t] = [0] * (cap + 1)
                if d:
                    for l in range(cap + 1):
                        w = row[l]
                        if w:
                            l2 = l + d
                            if l2 > cap:
                                l2 = cap
                            nr[l2] += w * wb
                else:
                    for l in range(cap + 1):
                        w = row[l]
                        if w:
                            nr[l] += w * wb
        cur = nxt
        if i in snap_ends:
            snaps[i] = [cur[p][0] if cur[p] else 0 for p in range(size)]
    total = 0
    dist = [0] * (cap + 1)
    for pat in range(size):
        row = cur[pat]
        if row:
            for l in range(cap + 1):
                if row[l]:
                    dist[l] += row[l]
                    total += row[l]
    return total, dist, snaps


def run_backward(n, K, bw, trans):
    """反向链式求和：back[j][pat] = 位置 j 之后（j+1..n-1）全部可行补全的权重和。

    与 run_dp 共用同一张转移表 trans（含质检置零），只是沿位置倒序递推、
    不带损失维。用于把位置 e 的前向模式质量补全为全分布边际：
    全边际(pat@e) = 前向质量[e][pat] × back[e][pat]。
    """
    size = 1 << K
    back = [None] * n
    back[n - 1] = [1] * size
    for j in range(n - 1, 0, -1):
        nxt = back[j]
        tj = trans[j]
        w0, w1 = bw[j]
        cur = [0] * size
        for pat in range(size):
            t0, t1 = tj[pat]
            s = 0
            if t0 >= 0:
                s += w0 * nxt[t0]
            if t1 >= 0:
                s += w1 * nxt[t1]
            cur[pat] = s
        back[j - 1] = cur
    return back


# ---------------------------------------------------------------- 主计算
def compute(model):
    """对给定模型做完整精确分析，返回中间结果字典（Fraction 级精确）。"""
    policy = model["policy"]
    area_min = int(policy["area_at_least"])
    thr = Fraction(policy["probability_at_least"])
    chain = build_chain(model)
    a_batches = chain["a_batches"]
    m = len(a_batches)
    b_order = chain["b_order"]
    n = chain["n"]
    K = chain["K"]
    tiles = chain["tiles"]
    pools = chain["pools"]
    total_area = sum(t["area"] for t in model["tiles"])
    cap_tail = total_area - area_min + 1      # 损失 <= total-area_min 即达标
    cap_pmf = total_area + 1                  # 完整分布不截断
    bw = []
    D_B = 1
    for b in b_order:
        nn, dd = model["priors"][b]
        bw.append((dd - nn, nn))
        D_B *= dd
    tabs = [tile_table(t["rule"], t["a"], t["b_names"]) for t in tiles]

    jobs = model["jobs"]
    job_ids = sorted(j["id"] for j in jobs)
    jcost = {j["id"]: j["cost"] for j in jobs}
    repairs = {j["id"]: set(j["repairs"]) for j in jobs}
    subsets = sorted((tuple(c) for r in range(len(job_ids) + 1)
                      for c in combinations(job_ids, r)), key=list)
    a_set = set(a_batches)
    b_index = chain["b_index"]
    subset_repair = {}
    for S in subsets:
        rep = set()
        for j in S:
            rep |= repairs[j]
        rb = 0
        for b in rep:
            if b in b_index:
                rb |= 1 << b_index[b]
        subset_repair[S] = (rep & a_set, rb)

    # ---- 每种 A 赋值：先验、池约束调整、转移表 ----
    aprior = {}
    a_info = {}
    for a_bits in range(1 << m):
        assign = {a_batches[i]: (a_bits >> i) & 1 for i in range(m)}
        f = Fraction(1)
        for name in a_batches:
            nn, dd = model["priors"][name]
            f *= Fraction(nn if assign[name] else dd - nn, dd)
        aprior[a_bits] = f
        pools_at = [[] for _ in range(n)]
        ok = True
        for p in pools:
            acnt = sum(assign[x] for x in p["a"])
            if p["blen"] == 0:
                if not p["lo"] <= acnt <= p["hi"]:
                    ok = False
                    break
                continue
            lo2 = max(0, p["lo"] - acnt)
            hi2 = min(p["blen"], p["hi"] - acnt)
            if lo2 > hi2:
                ok = False
                break
            pools_at[p["end"]].append((p["start"] - p["end"] + K - 1,
                                       (1 << p["blen"]) - 1, lo2, hi2))
        a_info[a_bits] = {"assign": assign,
                          "trans": build_trans(n, K, pools_at)} if ok else None

    tile_ends = {t["end"] for t in tiles if t["blen"] > 0}

    def make_tile_arrays(assign):
        """返回函数：给定重制子集 S，产出每块的 (端点,移位,掩码,损失表) 及常量损失。"""
        def tile_arrays(S):
            ra, rb_mask = subset_repair[S]
            out = []
            const_delta = 0
            for ti, t in enumerate(tiles):
                na, nb = len(t["a"]), t["blen"]
                a_bits_l = 0
                a_rep_l = 0
                for k, name in enumerate(t["a"]):
                    if assign[name]:
                        a_bits_l |= 1 << k
                    if name in ra:
                        a_rep_l |= 1 << k
                base = (a_bits_l << na) | a_rep_l
                if nb == 0:
                    out.append(("C", not tabs[ti][base]))
                    if not tabs[ti][base]:
                        const_delta += t["area"]
                    continue
                b_rep = (rb_mask >> t["start"]) & ((1 << nb) - 1)
                tab = tabs[ti]
                arr = [0] * (1 << nb)
                for wbv in range(1 << nb):
                    if not tab[(((base << nb) | wbv) << nb) | b_rep]:
                        arr[wbv] = t["area"]
                out.append(("W", t["end"], t["start"] - t["end"] + K - 1,
                            (1 << nb) - 1, arr))
            return out, const_delta
        return tile_arrays

    def delta_pos_of(tarrays):
        delta_pos = [None] * n
        for item in tarrays:
            if item[0] != "W":
                continue
            _, end, shift, vmask, arr = item
            dpos = delta_pos[end]
            if dpos is None:
                dpos = delta_pos[end] = [0] * (1 << K)
            for np_ in range(1 << K):
                dpos[np_] += arr[(np_ >> shift) & vmask]
        return delta_pos

    # ---- 完整性自检：无约束 DP 总质量必须恰为所有 den 之积（每个 B 世界恰计一次）----
    free_trans = build_trans(n, K, [[] for _ in range(n)])
    free_total, _, _ = run_dp(n, K, bw, free_trans, [None] * n, 0, 0)
    assert free_total == D_B, "chain DP lost or double-counted worlds"

    # ---- 逐 A 赋值跑 DP ----
    Z = {}
    um_a = {}        # a -> 每块可用质量（无重制）
    pmf_lost = {}    # a -> 完整损失分布
    tail_mass = {}   # (a, S) -> 达标质量
    snaps_a = {}
    tarr0_a = {}
    back_a = {}
    for a_bits, info in a_info.items():
        if info is None:
            continue
        assign = info["assign"]
        trans = info["trans"]
        tile_arrays = make_tile_arrays(assign)
        ta0, cd0 = tile_arrays(())
        total, _, snaps = run_dp(n, K, bw, trans, delta_pos_of(ta0), 0, cd0, tile_ends)
        if total == 0:
            continue  # 该 A 赋值与质检联合不可行，零质量剔除
        Z[a_bits] = total
        snaps_a[a_bits] = snaps
        tarr0_a[a_bits] = ta0
        um = []
        back = run_backward(n, K, bw, trans)
        for e in tile_ends:
            # 完备性交叉断言：任一位置的前向×反向必须还原总质量 Z
            assert sum(snaps[e][p] * back[e][p] for p in range(1 << K)) == total
        for ti, t in enumerate(tiles):
            item = ta0[ti]
            if item[0] == "C":
                um.append(0 if item[1] else total)
                continue
            _, end, shift, vmask, arr = item
            um.append(sum(mss * back[end][p] for p, mss in enumerate(snaps[end])
                          if mss and arr[(p >> shift) & vmask] == 0))
        um_a[a_bits] = um
        back_a[a_bits] = back
        total2, dist2, _ = run_dp(n, K, bw, trans, delta_pos_of(ta0), cap_pmf, cd0)
        assert total2 == total, "PMF DP weight not conserved"
        pmf_lost[a_bits] = dist2
        for S in subsets:
            taS, cdS = tile_arrays(S)
            totS, distS, _ = run_dp(n, K, bw, trans, delta_pos_of(taS), cap_tail, cdS)
            assert totS == total, "tail DP weight not conserved"
            tail_mass[(a_bits, S)] = sum(distS[:cap_tail])

    if not Z:
        raise ValueError("ZERO_EVIDENCE")
    W = {a: aprior[a] * Fraction(Z[a], D_B) for a in Z}
    P_E = sum(W.values(), Fraction())
    if P_E <= 0:
        raise ValueError("ZERO_EVIDENCE")

    # ---- 现状（不查证、不重制）----
    tile_usable = {}
    for ti, t in enumerate(tiles):
        tile_usable[t["id"]] = sum(W[a] * Fraction(um_a[a][ti], Z[a]) for a in Z) / P_E
    pmf = {}
    for a in Z:
        for l in range(cap_pmf):
            if pmf_lost[a][l]:
                area = total_area - l
                pmf[area] = pmf.get(area, Fraction()) + W[a] * Fraction(pmf_lost[a][l], Z[a]) / P_E
    tail_cur = sum((p for v, p in pmf.items() if v >= area_min), Fraction())
    tail_check = sum(W[a] * Fraction(tail_mass[(a, ())], Z[a]) for a in Z) / P_E
    assert tail_check == tail_cur, "current tail inconsistent"

    # ---- 相关性示例：两块共享血缘的联合可用概率 ----
    pair = None
    if len(tiles) >= 2 and tiles[0]["blen"] > 0 and tiles[1]["blen"] > 0:
        ti, tj = 0, 1
        e = max(tiles[ti]["end"], tiles[tj]["end"])
        if tiles[ti]["start"] >= e - K + 1 and tiles[tj]["start"] >= e - K + 1:
            shi = tiles[ti]["start"] - e + K - 1
            shj = tiles[tj]["start"] - e + K - 1
            mi = (1 << tiles[ti]["blen"]) - 1
            mj = (1 << tiles[tj]["blen"]) - 1
            num = Fraction()
            for a in Z:
                arri = tarr0_a[a][ti][4]
                arrj = tarr0_a[a][tj][4]
                s = 0
                for pat, mss in enumerate(snaps_a[a][e]):
                    if mss and arri[(pat >> shi) & mi] == 0 and arrj[(pat >> shj) & mj] == 0:
                        s += mss * back_a[a][e][pat]
                num += W[a] * Fraction(s, Z[a])
            pair = {"tiles": (tiles[ti]["id"], tiles[tj]["id"]),
                    "p_i": tile_usable[tiles[ti]["id"]],
                    "p_j": tile_usable[tiles[tj]["id"]],
                    "p_joint": num / P_E}

    # ---- 知识状态 ----
    all_codes = ["".join(c) for c in product("?01", repeat=m)]
    states = {}
    for code in all_codes:
        cons = [a for a in Z
                if all(code[i] == "?" or (a >> i) & 1 == int(code[i]) for i in range(m))]
        M = sum((W[a] for a in cons), Fraction())
        if M > 0:
            states[code] = {"M": M, "prob": M / P_E, "consistent": cons}
    for code, st in states.items():
        cons = st["consistent"]
        tden = sum(aprior[a] * Z[a] for a in cons)
        plans = []
        for S in subsets:
            tnum = sum(aprior[a] * tail_mass[(a, S)] for a in cons)
            plans.append({"jobs": list(S), "cost": sum(jcost[j] for j in S),
                          "tail": Fraction(tnum, tden)})
        st["plans"] = plans

    # ---- Bellman 递推（按 '?' 数从少到多，保证子状态先算）----
    insp = model["inspections"]
    for code in sorted(states, key=lambda c: c.count("?")):
        st = states[code]
        cands = []
        for pl in st["plans"]:
            if pl["tail"] >= thr:
                cands.append((Fraction(pl["cost"]), 0, pl["jobs"], {"stop": pl["jobs"]}))
        for k in range(m):
            if code[k] == "?":
                val = Fraction(insp[k]["cost"])
                for ans in "01":
                    child = code[:k] + ans + code[k+1:]
                    if child in states:
                        val += states[child]["M"] / st["M"] * states[child]["value"]
                cands.append((val, 1, [insp[k]["id"]], {"inspect": insp[k]["id"]}))
        cands.sort(key=lambda c: (c[0], c[1], c[2]))
        st["cands"] = cands
        st["value"] = cands[0][0]
        st["action"] = cands[0][3]

    state_order = sorted(states, key=lambda c: (m - c.count("?"), c))
    root = "?" * m
    assert abs(sum(states[c]["prob"] for c in states if "?" not in c) - 1) == 0
    assert states[root]["plans"][0]["tail"] == tail_cur

    # ---- 血缘共享统计 ----
    lineage = {b: {"pools": [], "tiles": []} for b in model["priors"]}
    for p in pools:
        for b in p["a"] + p["b_names"]:
            lineage[b]["pools"].append(p["id"])
    for t in tiles:
        for b in t["a"] + t["b_names"]:
            lineage[b]["tiles"].append(t["id"])

    a_table = []
    for a_bits in range(1 << m):
        assign = {a_batches[i]: (a_bits >> i) & 1 for i in range(m)}
        a_table.append({
            "bits": "".join(str(assign[x]) for x in a_batches),
            "assign": assign, "aprior": aprior[a_bits],
            "Z": Z.get(a_bits), "W": W.get(a_bits)})

    return {
        "model": model, "chain": chain, "thr": thr, "area_min": area_min,
        "total_area": total_area, "cap_tail": cap_tail, "D_B": D_B,
        "P_E": P_E, "W": W, "Z": Z, "aprior": aprior,
        "tile_usable": tile_usable, "pmf": pmf, "tail": tail_cur, "pair": pair,
        "states": states, "state_order": state_order, "root": root,
        "expected_cost": states[root]["value"],
        "unreachable": [c for c in all_codes if c not in states],
        "subsets": subsets, "jcost": jcost,
        "insp_ids": [x["id"] for x in insp],
        "insp_cost": {x["id"]: x["cost"] for x in insp},
        "lineage": lineage, "a_table": a_table,
        "free_total_ok": free_total == D_B,
    }


# ---------------------------------------------------------------- 结果渲染
def render_result(comp):
    model = comp["model"]
    states_out = {}
    for code in comp["state_order"]:
        st = comp["states"][code]
        states_out[code] = {
            "probability": fr(st["prob"]),
            "plans": [{"jobs": pl["jobs"], "cost": pl["cost"],
                       "tail_probability": fr(pl["tail"])} for pl in st["plans"]],
            "value": fr(st["value"]),
            "action": st["action"],
        }
    return {
        "evidence_probability": fr(comp["P_E"]),
        "current": {
            "tile_usable": {t["id"]: fr(comp["tile_usable"][t["id"]])
                            for t in model["tiles"]},
            "area_pmf": [[v, fr(p)] for v, p in sorted(comp["pmf"].items()) if p > 0],
            "tail_probability": fr(comp["tail"]),
        },
        "states": states_out,
        "expected_cost": fr(comp["expected_cost"]),
    }


# ---------------------------------------------------------------- 修订台账
def revision_ledger():
    """逐行解析 pools.jsonl，给出每行的处置（生效/取代/重放/撤销）。"""
    raw_lines = []
    for ln, raw in enumerate((ROOT / "data/pools.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            raw_lines.append((ln, json.loads(raw)))
    by_id = {}
    for ln, row in raw_lines:
        by_id.setdefault(row["id"], []).append((ln, row))
    ledger = []
    for pid in sorted(by_id):
        entries = by_id[pid]
        win_ln, win = max(entries, key=lambda e: e[1]["rev"])
        for ln, row in entries:
            if ln == win_ln:
                disp = "撤销生效：该 id 不约束" if row["deleted"] else "生效"
            elif row["rev"] == win["rev"]:
                disp = "同版同内容重放：忽略"
            else:
                disp = f"被 rev{win['rev']}（第 {win_ln} 行）取代"
            ledger.append({"line": ln, "id": pid, "rev": row["rev"],
                           "deleted": row["deleted"], "lo": row["lo"], "hi": row["hi"],
                           "disp": disp, "active": ln == win_ln and not row["deleted"]})
    return ledger


# ---------------------------------------------------------------- 报告
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def action_str(act):
    if "stop" in act:
        return "结束，重制 " + ("+".join(act["stop"]) if act["stop"] else "∅（不重制）")
    return "查证 " + act["inspect"]


def policy_tree_lines(comp, code, depth, out):
    st = comp["states"][code]
    pad = "  " * depth
    act = st["action"]
    if "stop" in act:
        jobs = act["stop"]
        tail = next(pl["tail"] for pl in st["plans"] if pl["jobs"] == jobs)
        out.append(f"{pad}- 状态 `{code}`：结束，重制 "
                   f"{'+'.join(jobs) if jobs else '∅'}，后续费用 {fr(st['value'])}，"
                   f"该终点达标概率 {fr(tail)}")
    else:
        iid = act["inspect"]
        k = comp["insp_ids"].index(iid)
        out.append(f"{pad}- 状态 `{code}`：查证 {iid}（费 {comp['insp_cost'][iid]}），"
                   f"该状态价值 {fr(st['value'])}")
        for ans in "01":
            child = code[:k] + ans + code[k+1:]
            if child in comp["states"]:
                p = comp["states"][child]["M"] / st["M"]
                out.append(f"{pad}  - 答复 {iid}={ans}（分支概率 {fr(p)}）：")
                policy_tree_lines(comp, child, depth + 2, out)


def render_report(comp, ledger):
    model = comp["model"]
    chain = comp["chain"]
    L = []
    A = L.append
    A("# 封存批覆盖质量核定报告")
    A("")
    A("本报告由 `python3 analysis.py run` 从 `data/` 原始材料重算生成；"
      "全部为精确有理数结果，不枚举全部缺陷赋值、不做独立块近似、不做蒙特卡洛。")
    A("")
    # ------------------------------------------------ 1 材料与修订
    A("## 1. 登记材料与修订台账")
    A("")
    pr = model["priors"]
    A(f"- `data/batches.csv`：{len(pr)} 个源片批，先验相互独立；"
      f"A0={fr(Fraction(*pr['A0']))}、A1={fr(Fraction(*pr['A1']))}、A2={fr(Fraction(*pr['A2']))} 为缺陷先验，"
      "B00..B49 按 (1/10, 2/11, 3/12, 1/13) 循环。每个批号的缺陷状态只有一份，"
      "被所有质检与血缘共享（同源缺陷始终共享）。")
    A(f"- `data/mosaic.json`：{len(model['tiles'])} 块成果，总面积 {comp['total_area']} 平方公里；"
      f"门槛：可用面积 ≥ {comp['area_min']} 的概率 ≥ {fr(comp['thr'])}。")
    A("- `data/rework.json`：6 项重制任务 "
      + "、".join(f"{j['id']}(费{j['cost']})" for j in model["jobs"])
      + f"，全做费用 {sum(j['cost'] for j in model['jobs'])} 必使 60 平方公里全可用；"
      + "3 项查证 " + "、".join(f"{x['id']}→{x['batch']}(费{x['cost']})" for x in model["inspections"]) + "。")
    A("")
    A("`data/pools.jsonl` 共 20 行原始记录。按 id 取最高 rev，撤销胜者不约束，"
      "同 id 同 rev 异内容即冲突（本批无冲突），完全同内容重放忽略。逐行处置：")
    A("")
    rows = [(e["line"], e["id"], e["rev"], "是" if e["deleted"] else "否",
             f"{e['lo']}..{e['hi']}", e["disp"]) for e in ledger]
    A("\n".join(md_table(["行号", "id", "rev", "deleted", "lo..hi", "处置"], rows)))
    A("")
    active = [e for e in ledger if e["active"]]
    A(f"有效质检共 {len(active)} 条（与 `checks.py` 的回归断言一致）。要点："
      "P04 的 rev2（2..3，第 5 行）取代 rev1（0..0，第 20 行）；"
      "P07 的 rev2（第 10 行）带撤销标志，故 P07 整体不约束；"
      "P02 的两次 rev1（第 3、19 行）内容完全相同，属重放，只计一次。")
    A("")
    A("有效质检在 B 链（B00..B49）上的窗口与缺陷计数约束：")
    A("")
    rows = []
    for p in chain["pools"]:
        rows.append((p["id"], "+".join(p["a"]) if p["a"] else "—",
                     f"{p['b_names'][0]}..{p['b_names'][-1]}" if p["b_names"] else "—",
                     f"{p['lo']}..{p['hi']}"))
    A("\n".join(md_table(["质检", "涉及 A 批", "B 链窗口", "缺陷计数"], rows)))
    A("")
    # ------------------------------------------------ 2 证据与条件化
    A("## 2. 证据概率与联合条件化")
    A("")
    A("全部原质检事件记为 E。三个可查证批号 A0、A1、A2 同时出现在多条质检与多块血缘中，"
      "是仅有的长程变量；先对它们的 8 种赋值联合条件化。给定 A 赋值后，B 链上只剩"
      "连续窗口约束，用链式 DP 精确求和（见第 4 节）。先验不因条件化而因子化——"
      "后验始终保持联合，绝不由块边际独立拼成。")
    A("")
    A(f"**P(E) = {fr(comp['P_E'])}**（精确值）。各 A 赋值的质量分解：")
    A("")
    rows = []
    for row in comp["a_table"]:
        if row["Z"] is None:
            rows.append((row["bits"], fr(row["aprior"]), "不可行（违反质检）", "—", "0/1"))
        else:
            rows.append((row["bits"], fr(row["aprior"]), str(row["Z"]), fr(row["W"]),
                         fr(row["W"] / comp["P_E"])))
    A("\n".join(md_table(["A0A1A2 缺陷位", "A 先验权重", "B 链整数质量 Z", "联合权重 W", "后验 P(赋值|E)"], rows)))
    A("")
    # ------------------------------------------------ 3 现状
    A("## 3. 现状（只做原质检，不查证不重制）")
    A("")
    A("各成果块可用概率 P(块可用 | E)：")
    A("")
    rows = [(t["id"], t["area"], fr(comp["tile_usable"][t["id"]])) for t in model["tiles"]]
    A("\n".join(md_table(["成果块", "面积", "可用概率"], rows)))
    A("")
    A("当前可用总面积的完整分布（只列正质量）：")
    A("")
    rows = [(v, fr(p)) for v, p in sorted(comp["pmf"].items()) if p > 0]
    A("\n".join(md_table(["可用面积", "概率"], rows)))
    A("")
    A(f"**现状达标概率 P(面积 ≥ {comp['area_min']} | E) = {fr(comp['tail'])}**，"
      f"低于门槛 {fr(comp['thr'])}，必须采取行动（查证和/或重制）。")
    A("")
    if comp["pair"]:
        pr_ = comp["pair"]
        A("### 3.1 为什么现状分布不能由块边际概率独立拼成")
        A("")
        A(f"以 {pr_['tiles'][0]}、{pr_['tiles'][1]} 为例：两者血缘共享批号 B02、B03 与 A1"
          "（T00 规则含 B00..B03、A0、A1；T01 含 B02..B05、A1、A2），"
          "且质检 P00、P01 把同一批号耦进同一计数约束。精确计算给出：")
        A("")
        A(f"- P({pr_['tiles'][0]} 可用) = {fr(pr_['p_i'])}，P({pr_['tiles'][1]} 可用) = {fr(pr_['p_j'])}")
        A(f"- 独立乘积 = {fr(pr_['p_i'] * pr_['p_j'])}")
        A(f"- 实际联合 P(两块同可用) = {fr(pr_['p_joint'])} ≠ 独立乘积")
        A("")
        A(f"此外每个 A 批同时出现在 {len(comp['lineage']['A0']['tiles'])} 块规则与多条质检中，是全域公共原因；"
          "对 A 边际化后各块仍然相关。因此面积分布不是各块边际的独立卷积，"
          "必须沿留存血缘联合条件化——本报告用链式 DP 做了精确联合求值。")
        A("")
    # ------------------------------------------------ 4 方法
    A("## 4. 精确推断方法：链式 DP、状态合并与完整性")
    A("")
    A("### 4.1 链结构与合并状态")
    A("")
    A("条件化 A 赋值后，每条有效质检是 B 链上长度 ≤ 6 的连续窗口计数约束，"
      "每块成果的血缘窗口也是长度 ≤ 4 的连续区间（见第 1 节窗口表）。"
      "沿 B00..B49 逐位推进，DP 状态只保留：")
    A("")
    A("- **最近 K=6 位缺陷模式**：任何尚未完成的质检/血缘窗口（长度 ≤ K）读取的"
      "批号都在这 K 位之内，更早的历史对未来约束与血缘求值毫无影响；")
    A("- **截断损失面积**：已完成的不可用块面积之和。达标判定只问总损失是否 "
      f"≤ {comp['total_area']}−{comp['area_min']}={comp['cap_tail'] - 1}，"
      f"故尾部分析把损失截断在 {comp['cap_tail']}（≥该值一律归入失败桶）；"
      "完整分布则用不截断的 0..60 精确记账。")
    A("")
    A("转移质量 = 当前 B 批的先验整数权重（良好 den−n、缺陷 n；公共分母 Π den "
      "在最后统一约掉，全程整数/有理数精确运算）。质检窗口在完成位置校验计数，"
      "违反的转移直接置零（零质量）；成果块在窗口完成位置记账：不可用则累加其面积。")
    A("")
    A("### 4.2 状态合并不丢相关性（证明）")
    A("")
    A("设两段历史 H1、H2 在同一位置具有相同的 (最近 K 位模式, 截断损失)。"
      "则：(i) 未来任何窗口只读最近 K 位，二者面对完全相同的可行转移集；"
      "(ii) 未来每位转移质量只取决于该位自身先验，与更早历史无关；"
      "(iii) 损失只加不减，截断只把 ≥cap 的值合并，而这些值在任何未来都仍 ≥cap，"
      "达标/失败的分类不变。故 H1、H2 对未来一切事件（约束满足、各块可用、损失分布）"
      "的加权测度完全相同，合并它们不丢任何相关性。A 批之间的相关、A 与 B 的相关"
      "由 8 路联合条件化完整保留；B 链内部的相关由 K 位模式完整保留。"
      "同源缺陷始终共享：每个批号在 DP 中只占一位，所有引用它的质检与血缘读同一位；"
      "查证只是揭示该位，重制只替换血缘规则中的质量输入，二者都不改写原缺陷世界。")
    A("")
    A("### 4.3 完整性与零概率状态")
    A("")
    A(f"完备性：对同一链不做任何质检约束跑同一 DP，总质量精确等于 Π den"
      f"（本批为 {comp['D_B']}），即 2^50 个 B 世界每个恰好计一次、权重精确；"
      "加入质检只是把违反世界的转移置零，不重不漏。该断言在每次重算时自动核验"
      f"（本次：{'通过' if comp['free_total_ok'] else '失败'}）。"
      "知识状态只收 P(答复 | E) > 0 者；零概率答复不属于任何状态、不对它条件化、"
      "策略也不执行该分支。")
    A("")
    # ------------------------------------------------ 5 状态
    A("## 5. 全部可达知识状态的后验权重")
    A("")
    A("状态编码三字符，位置依查证文件顺序（I0→A0、I1→A1、I2→A2），? 未查、0 良好、1 有缺陷。"
      "probability 是 P(该部分答复 | E)，不是不同答复间的归一化权重。")
    A("")
    rows = [(c, fr(comp["states"][c]["prob"])) for c in comp["state_order"]]
    A("\n".join(md_table(["状态", "P(答复 | E)"], rows)))
    A("")
    if comp["unreachable"]:
        A("零概率（不可达、不列入策略）状态：" + "、".join(f"`{c}`" for c in comp["unreachable"]) + "。")
    else:
        A("全部 27 个部分答复状态概率均为正，均可达。")
    A("")
    # ------------------------------------------------ 6 根状态组合表
    A("## 6. 各重制组合的达标概率")
    A("")
    A("根状态 `???`（未做任何查证）下全部 64 个重制子集的达标概率"
      "（其余每个状态的 64 行全表见附录 B，机器可读版见 result.json 的 plans）：")
    A("")
    rows = [("+".join(pl["jobs"]) if pl["jobs"] else "∅", pl["cost"], fr(pl["tail"]))
            for pl in comp["states"][comp["root"]]["plans"]]
    A("\n".join(md_table(["重制组合", "费用", "达标概率"], rows)))
    A("")
    # ------------------------------------------------ 7 策略
    A("## 7. 最优策略、动作价值与全局最优性")
    A("")
    A("设 V(s) 为状态 s 起的最小期望后续费用（不含已付查证费）。候选动作：")
    A("")
    A("- 结束并执行重制子集 S：仅当 P(面积 ≥ 门槛 | s, S) ≥ " + fr(comp["thr"])
      + " 时合法（每个正概率终点必须单独达标），价值 = 费用(S)；")
    A("- 查证第 k 项（未查过）：价值 = 查证费 + Σ_答复 P(答复 | s) · V(子状态)，"
      "只取正概率答复分支。")
    A("")
    A("V(s) = 所有候选动作价值的最小值；平局时先结束后查证，同类按动作 ID 序列的"
      "Python 字符串字典序取首（结束动作的任务 ID 先排序，查证动作序列仅含查证 ID）。")
    A("")
    A("各状态全部候选动作的条件期望与选定动作（“最优结束”只列费用最小的合法组合，"
      "其余合法组合费用更高、被支配）:")
    A("")
    rows = []
    for code in comp["state_order"]:
        st = comp["states"][code]
        stops = [c for c in st["cands"] if c[1] == 0]
        insps = [c for c in st["cands"] if c[1] == 1]
        best_stop = stops[0] if stops else None
        if best_stop:
            tail = next(pl["tail"] for pl in st["plans"] if pl["jobs"] == best_stop[2])
            stop_txt = f"{'+'.join(best_stop[2]) or '∅'} 费{fr(best_stop[0])} 达标{fr(tail)}"
        else:
            stop_txt = "无合法组合"
        insp_txt = "；".join(f"{c[2][0]}:{fr(c[0])}" for c in insps) or "—"
        rows.append((f"`{code}`", stop_txt, insp_txt, action_str(st["action"]), fr(st["value"])))
    A("\n".join(md_table(["状态", "最优结束(组合/费用/达标)", "各查证条件期望", "选定动作", "V(s)"], rows)))
    A("")
    A(f"**最优期望总费用 = V(???) = {fr(comp['expected_cost'])}**（含沿途查证费与终点重制费）。")
    A("")
    A("### 7.1 全局最优性证明")
    A("")
    A("状态按已查项数构成有限 DAG（查证只增不减），任意合法策略 π 从状态 s 出发的"
      "期望费用 C_π(s) 必满足同一递推：若 π 在 s 结束于组合 S，则 C_π(s)=费用(S)；"
      "若 π 在 s 查证 k，则 C_π(s)=查证费+Σ P(答复|s)·C_π(子状态)。"
      "对未查项数作归纳：基础情形（三项均已查）只剩结束动作，V(s) 是合法组合的最小费用，"
      "显然 ≤ 任何 π；归纳情形，若 π 结束，C_π(s) 是某个合法组合费用 ≥ 最小者 ≥ V(s)；"
      "若 π 查证 k，由归纳假设每个子状态 C_π ≥ V，故 C_π(s) ≥ 该查证动作的候选价值 ≥ V(s)。"
      "于是 V(s) ≤ C_π(s) 对一切 π 成立，而按上表选定动作构造的策略恰取得 V(s)，"
      "故为全局最优。每个正概率终点在选定前已强制满足达标门槛，"
      "策略不可能依赖未查答案（动作只是当前状态的函数），零概率分支不执行。")
    A("")
    # ------------------------------------------------ 8 分支
    A("## 8. 策略分支决策（仅正概率答复）")
    A("")
    tree = []
    policy_tree_lines(comp, comp["root"], 0, tree)
    A("\n".join(tree))
    A("")
    # ------------------------------------------------ 9 附录A 血缘
    A("## 附录 A. 共享血缘台账")
    A("")
    A("每个批号被多少条质检、多少块成果引用（共享越多的批号越能解释块间相关）：")
    A("")
    rows = []
    for b in sorted(comp["lineage"]):
        info = comp["lineage"][b]
        if b.startswith("A") or len(info["tiles"]) >= 2:
            rows.append((b, "+".join(info["pools"]), len(info["tiles"]),
                         "+".join(info["tiles"])))
    A("\n".join(md_table(["批号", "所属质检", "引用块数", "引用块"], rows)))
    A("")
    A("（B 批中仅列被 ≥2 块引用的；相邻成果块窗口错位 2 位、长 4 位，"
      "恒有 2 个 B 批重叠，故任意相邻块必共享血缘。）")
    A("")
    # ------------------------------------------------ 附录B 全状态组合表
    A("## 附录 B. 各状态全部重制组合的达标概率")
    A("")
    for code in comp["state_order"]:
        st = comp["states"][code]
        A(f"### 状态 `{code}`（P={fr(st['prob'])}）")
        A("")
        rows = [("+".join(pl["jobs"]) if pl["jobs"] else "∅", pl["cost"], fr(pl["tail"]))
                for pl in st["plans"]]
        A("\n".join(md_table(["重制组合", "费用", "达标概率"], rows)))
        A("")
    # ------------------------------------------------ 10 复核
    A("## 复核与一致性")
    A("")
    A("`python3 analysis.py verify` 执行以下核对，任一不符即非零退出且不改写产物：")
    A("")
    A("1. 重算本批并与磁盘上的 result.json 全等比对；REPORT.md 须含关键数值；")
    A("2. 运行原回归 `checks.py`（不动 source.py、checks.py、data/ 及现有断言）；")
    A("3. 用 `checks.fixture()` 的三批号小例做独立穷举判定：判定器直接枚举缺陷世界，"
      "不调用分析器的状态合并或策略求解函数，逐项比对后验分布、条件分支与最优费用"
      "（小例：后验面积 {0: 1/4, 3: 3/4}，查证分支各 1/2，最优期望费用 4——"
      "查证 a 后按答复重制对应批号）；")
    A("4. 仅用结果文件中的状态与动作，独立重推：每个结束动作的达标概率 ≥ 门槛、"
      "每个状态两个答复分支的概率之和 = 该状态概率、选定动作是全部候选动作的"
      "（价值, 类别, ID 序列）最小者、状态价值满足费用递推、根价值 = expected_cost、"
      "现状尾部概率 = 根状态空组合达标概率、面积分布归一、完整答复状态概率和 = 1；")
    A("5. 链式 DP 完整性断言（无约束总质量 = Π den）与权重守恒断言"
      "（每个 (A 赋值, 重制组合) 的 DP 总质量 = 该赋值的 Z）随每次重算自动执行。")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 独立判定器
def judge(model):
    """小例直接枚举判定器：枚举全部缺陷世界，独立重算后验与策略。

    只调用 source.py 的既有语义函数，不调用本文件的分析器（链式合并 DP 与
    策略求解）函数；用于与分析器在小例上互核。
    """
    names = sorted(model["priors"])
    if len(names) > 24:
        raise ValueError("JUDGE_MODEL_TOO_BIG")
    worlds = []
    for r in range(len(names) + 1):
        for combo in combinations(names, r):
            fs = set(combo)
            if source.evidence_ok(model, fs):
                worlds.append((fs, source.weight(model, fs)))
    mass = sum((w for _, w in worlds), Fraction())
    if mass <= 0:
        raise ValueError("ZERO_EVIDENCE")
    policy = model["policy"]
    area_min = int(policy["area_at_least"])
    thr = Fraction(policy["probability_at_least"])
    insp = model["inspections"]
    m = len(insp)
    job_ids = sorted(j["id"] for j in model["jobs"])
    jcost = {j["id"]: j["cost"] for j in model["jobs"]}
    subsets = sorted((tuple(c) for r in range(len(job_ids) + 1)
                      for c in combinations(job_ids, r)), key=list)

    def consistent(fs, code):
        return all(code[k] == "?" or
                   ((insp[k]["batch"] in fs) == (code[k] == "1")) for k in range(m))

    tile_usable = {}
    for t in model["tiles"]:
        tile_usable[t["id"]] = sum((w for fs, w in worlds
                                    if source.good(t["rule"], fs, ())), Fraction()) / mass
    pmf = {}
    for fs, w in worlds:
        area = source.evaluate(model, fs)["area"]
        pmf[area] = pmf.get(area, Fraction()) + w / mass
    tail = sum((p for v, p in pmf.items() if v >= area_min), Fraction())

    states = {}
    for code in map("".join, product("?01", repeat=m)):
        M = sum((w for fs, w in worlds if consistent(fs, code)), Fraction())
        if M <= 0:
            continue
        plans = []
        for S in subsets:
            num = sum((w for fs, w in worlds
                       if consistent(fs, code)
                       and source.evaluate(model, fs, S)["area"] >= area_min), Fraction())
            plans.append({"jobs": list(S), "cost": sum(jcost[j] for j in S),
                          "tail": num / M})
        states[code] = {"M": M, "prob": M / mass, "plans": plans}

    values, actions = {}, {}

    def solve(code):
        if code in values:
            return values[code]
        st = states[code]
        cands = []
        for pl in st["plans"]:
            if pl["tail"] >= thr:
                cands.append((Fraction(pl["cost"]), 0, pl["jobs"], {"stop": pl["jobs"]}))
        for k in range(m):
            if code[k] == "?":
                val = Fraction(insp[k]["cost"])
                for ans in "01":
                    child = code[:k] + ans + code[k+1:]
                    if child in states:
                        val += states[child]["M"] / st["M"] * solve(child)
                cands.append((val, 1, [insp[k]["id"]], {"inspect": insp[k]["id"]}))
        cands.sort(key=lambda c: (c[0], c[1], c[2]))
        values[code], actions[code] = cands[0][0], cands[0][3]
        return values[code]

    for code in states:
        solve(code)
    root = "?" * m
    out_states = {}
    for code in sorted(states, key=lambda c: (m - c.count("?"), c)):
        st = states[code]
        out_states[code] = {
            "probability": fr(st["prob"]),
            "plans": [{"jobs": pl["jobs"], "cost": pl["cost"],
                       "tail_probability": fr(pl["tail"])} for pl in st["plans"]],
            "value": fr(values[code]),
            "action": actions[code],
        }
    return {
        "evidence_probability": fr(mass),
        "current": {
            "tile_usable": {t["id"]: fr(tile_usable[t["id"]]) for t in model["tiles"]},
            "area_pmf": [[v, fr(p)] for v, p in sorted(pmf.items()) if p > 0],
            "tail_probability": fr(tail),
        },
        "states": out_states,
        "expected_cost": fr(values[root]),
    }


# ---------------------------------------------------------------- 合成小例
def synthetic_models():
    """结构更丰富的可穷举小例，供判定器与分析器互核（不触及本批数据）。"""
    m1 = {  # A-only 质检锁定 A0=1 -> 查证答复 0 为零概率分支；atleast/any/all 混合
        "priors": {"A0": (1, 3), "B0": (1, 2), "B1": (1, 3), "B2": (2, 5),
                   "B3": (1, 4), "B4": (3, 7), "B5": (1, 2), "B6": (2, 3), "B7": (1, 6)},
        "pools": [
            {"id": "q0", "rev": 1, "deleted": False, "batches": ["A0", "B0", "B1", "B2"], "lo": 1, "hi": 1},
            {"id": "q1", "rev": 1, "deleted": False, "batches": ["B2", "B3", "B4", "B5"], "lo": 2, "hi": 2},
            {"id": "q2", "rev": 1, "deleted": False, "batches": ["A0", "B5", "B6", "B7"], "lo": 0, "hi": 1},
            {"id": "q3", "rev": 1, "deleted": False, "batches": ["A0"], "lo": 1, "hi": 1},
        ],
        "tiles": [
            {"id": "x0", "area": 2, "rule": ["all", "A0", "B0", ["atleast", 2, "B1", "B2", "B3"]]},
            {"id": "x1", "area": 3, "rule": ["any", ["all", "B3", "B4"], ["atleast", 2, "B5", "B6", "B7"]]},
            {"id": "x2", "area": 1, "rule": ["atleast", 1, "B6", "B7"]},
        ],
        "jobs": [{"id": "J0", "cost": 2, "repairs": ["A0", "B0", "B1"]},
                 {"id": "J1", "cost": 3, "repairs": ["B4", "B5", "B6"]},
                 {"id": "J2", "cost": 1, "repairs": ["B7"]}],
        "inspections": [{"id": "I0", "batch": "A0", "cost": 1}],
        "policy": {"area_at_least": 4, "probability_at_least": "19/20"},
    }
    m2 = {  # 两个 A；A0=0 的赋值与质检联合不可行（Z=0 剔除）
        "priors": {"A0": (1, 2), "A1": (1, 2), "B0": (1, 2), "B1": (1, 2),
                   "B2": (1, 2), "B3": (1, 2), "B4": (1, 2), "B5": (1, 2)},
        "pools": [
            {"id": "q0", "rev": 1, "deleted": False, "batches": ["A0", "B0", "B1"], "lo": 2, "hi": 3},
            {"id": "q1", "rev": 1, "deleted": False, "batches": ["A1", "B0", "B1"], "lo": 0, "hi": 1},
        ],
        "tiles": [
            {"id": "y0", "area": 2, "rule": ["any", "A0", "B0"]},
            {"id": "y1", "area": 3, "rule": ["all", "A1", ["any", "B1", "B2"]]},
            {"id": "y2", "area": 1, "rule": ["atleast", 2, "B3", "B4", "B5"]},
        ],
        "jobs": [{"id": "K0", "cost": 2, "repairs": ["A0", "B0"]},
                 {"id": "K1", "cost": 2, "repairs": ["A1", "B1", "B2"]}],
        "inspections": [{"id": "I0", "batch": "A0", "cost": 1},
                        {"id": "I1", "batch": "A1", "cost": 1}],
        "policy": {"area_at_least": 5, "probability_at_least": "19/20"},
    }
    m3 = {  # 同一位置多条质检窗口收尾；A-only 成果块（常量损失路径）
        "priors": {"A0": (1, 3), "B0": (2, 5), "B1": (1, 2), "B2": (1, 4),
                   "B3": (3, 5), "B4": (1, 2)},
        "pools": [
            {"id": "q0", "rev": 1, "deleted": False, "batches": ["A0", "B0", "B1", "B2"], "lo": 1, "hi": 2},
            {"id": "q1", "rev": 1, "deleted": False, "batches": ["B1", "B2"], "lo": 1, "hi": 1},
            {"id": "q2", "rev": 1, "deleted": False, "batches": ["B2", "B3", "B4"], "lo": 1, "hi": 3},
        ],
        "tiles": [
            {"id": "z0", "area": 4, "rule": ["any", ["all", "A0", ["atleast", 2, "B0", "B1", "B2"]],
                                             ["all", "B3", "B4"]]},
            {"id": "z1", "area": 2, "rule": ["all", "A0"]},
        ],
        "jobs": [{"id": "M0", "cost": 3, "repairs": ["A0"]},
                 {"id": "M1", "cost": 2, "repairs": ["B3", "B4"]}],
        "inspections": [{"id": "I0", "batch": "A0", "cost": 1}],
        "policy": {"area_at_least": 5, "probability_at_least": "9/10"},
    }
    m4 = {  # 平局规则：{J0,J1} 与 {J2} 同费 -> 字典序取前者；结束与查证同值 -> 先结束
        "priors": {"A0": (1, 2), "B0": (1, 2)},
        "pools": [],
        "tiles": [
            {"id": "w0", "area": 2, "rule": ["all", "A0", "B0"]},
        ],
        "jobs": [{"id": "J0", "cost": 2, "repairs": ["A0"]},
                 {"id": "J1", "cost": 2, "repairs": ["B0"]},
                 {"id": "J2", "cost": 4, "repairs": ["A0", "B0"]}],
        "inspections": [{"id": "I0", "batch": "A0", "cost": 1}],
        "policy": {"area_at_least": 2, "probability_at_least": "3/4"},
    }
    return [("synthetic-Aonly-pool-zero-branch", m1),
            ("synthetic-infeasible-assignment", m2),
            ("synthetic-shared-endpoint-const-tile", m3),
            ("synthetic-tie-break", m4)]


# ---------------------------------------------------------------- 产物一致性核对
def check_result(res, model):
    """只用结果文件与原始模型，独立重推门槛、分支权重、动作选择与费用递推。"""
    thr = Fraction(model["policy"]["probability_at_least"])
    area_min = int(model["policy"]["area_at_least"])
    jcost = {j["id"]: j["cost"] for j in model["jobs"]}
    insp = model["inspections"]
    m = len(insp)
    icost = {x["id"]: x["cost"] for x in insp}
    F = Fraction
    states = res["states"]
    assert "?" * m in states, "root state missing"
    for code, st in states.items():
        plans = st["plans"]
        assert len(plans) == 2 ** len(jcost), f"{code}: plans count"
        jl = [p["jobs"] for p in plans]
        assert jl == sorted(jl), f"{code}: plans not sorted"
        for p in plans:
            assert p["jobs"] == sorted(p["jobs"]), f"{code}: jobs not sorted"
            assert p["cost"] == sum(jcost[j] for j in p["jobs"]), f"{code}: plan cost"
            assert F(0) <= F(p["tail_probability"]) <= 1, f"{code}: tail range"
    # 分支权重：任一未查位的两个答复子状态概率之和 = 本状态概率
    for code, st in states.items():
        for k in range(m):
            if code[k] == "?":
                tot = sum((F(states[c]["probability"])
                           for c in (code[:k] + a + code[k+1:] for a in "01")
                           if c in states), F())
                assert tot == F(st["probability"]), f"{code}: branch weights"
    # 动作选择与费用递推
    for code, st in states.items():
        cands = []
        for p in st["plans"]:
            if F(p["tail_probability"]) >= thr:
                cands.append((F(p["cost"]), 0, p["jobs"], {"stop": p["jobs"]}))
        assert cands, f"{code}: no feasible stop"
        for k, x in enumerate(insp):
            if code[k] == "?":
                val = F(icost[x["id"]])
                for a in "01":
                    c = code[:k] + a + code[k+1:]
                    if c in states:
                        val += F(states[c]["probability"]) / F(st["probability"]) * F(states[c]["value"])
                cands.append((val, 1, [x["id"]], {"inspect": x["id"]}))
        cands.sort(key=lambda c: (c[0], c[1], c[2]))
        assert F(st["value"]) == cands[0][0], f"{code}: value recursion"
        assert st["action"] == cands[0][3], f"{code}: action choice"
        if "stop" in st["action"]:
            pl = next(p for p in st["plans"] if p["jobs"] == st["action"]["stop"])
            assert F(pl["tail_probability"]) >= thr, f"{code}: terminal threshold"
    root = "?" * m
    assert F(states[root]["probability"]) == 1, "root probability"
    assert F(res["expected_cost"]) == F(states[root]["value"]), "expected_cost"
    plan0 = next(p for p in states[root]["plans"] if p["jobs"] == [])
    assert F(res["current"]["tail_probability"]) == F(plan0["tail_probability"]), "current tail"
    tot = sum((F(p) for _, p in res["current"]["area_pmf"]), F())
    assert tot == 1, "area pmf not normalized"
    tail2 = sum((F(p) for a, p in res["current"]["area_pmf"] if a >= area_min), F())
    assert tail2 == F(res["current"]["tail_probability"]), "pmf tail mismatch"
    assert set(res["current"]["tile_usable"]) == {t["id"] for t in model["tiles"]}, "tiles"
    full = [c for c in states if "?" not in c]
    assert sum((F(states[c]["probability"]) for c in full), F()) == 1, "full states sum"


def diff(a, b, path="$"):
    if type(a) is not type(b):
        yield f"{path}: {a!r} != {b!r}"
        return
    if isinstance(a, dict):
        for k in a.keys() | b.keys():
            if k not in a:
                yield f"{path}.{k}: missing in recomputed"
            elif k not in b:
                yield f"{path}.{k}: missing in product"
            else:
                yield from diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            yield f"{path}: length {len(a)} != {len(b)}"
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from diff(x, y, f"{path}[{i}]")
    elif a != b:
        yield f"{path}: {a!r} != {b!r}"


# ---------------------------------------------------------------- 入口
def cmd_run():
    try:
        model = source.load()
        comp = compute(model)
    except ValueError as exc:
        print(f"run 中止：{exc}（不写任何产物）", file=sys.stderr)
        return 1
    res = render_result(comp)
    report = render_report(comp, revision_ledger())
    (ROOT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"P(E)={fr(comp['P_E'])}")
    print(f"现状达标概率={fr(comp['tail'])}（门槛 {fr(comp['thr'])}）")
    print(f"最优期望总费用={fr(comp['expected_cost'])}")
    print("已写入 result.json 与 REPORT.md")
    return 0


def cmd_verify():
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print("VERIFY_FAIL: " + msg)

    try:
        model = source.load()
        comp = compute(model)
    except ValueError as exc:
        print(f"VERIFY_FAIL: 重算中止：{exc}")
        return 1
    rendered = json.loads(json.dumps(render_result(comp)))

    rj = ROOT / "result.json"
    disk = None
    if not rj.exists():
        fail("result.json 不存在")
    else:
        disk = json.loads(rj.read_text(encoding="utf-8"))
        if disk != rendered:
            bad = next(diff(rendered, disk), "unknown")
            fail("result.json 与重算不符：" + bad)
    rp = ROOT / "REPORT.md"
    if not rp.exists():
        fail("REPORT.md 不存在")
    else:
        text = rp.read_text(encoding="utf-8")
        for key in (fr(comp["P_E"]), fr(comp["expected_cost"]), fr(comp["tail"])):
            if key not in text:
                fail(f"REPORT.md 缺少关键数值 {key}")
    if disk is not None:
        try:
            check_result(disk, model)
        except AssertionError as exc:
            fail("产物一致性：" + str(exc))

    proc = subprocess.run([sys.executable, "checks.py"], cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        fail("原回归 checks.py 失败：" + proc.stdout + proc.stderr)

    import checks
    fixture = checks.fixture()
    cases = [("fixture", fixture)] + synthetic_models()
    for name, fmodel in cases:
        comp_f = compute(fmodel)
        rendered_f = json.loads(json.dumps(render_result(comp_f)))
        judged = judge(fmodel)
        if rendered_f != judged:
            bad = next(diff(rendered_f, judged), "unknown")
            fail(f"小例互核不符（{name}）：" + bad)

    if ok:
        print("重算与产物一致；原回归通过；小例穷举互核通过；产物一致性通过。")
        print("VERIFY_OK")
        return 0
    return 1


def cmd_demo():
    model = source.load()
    comp = compute(model)
    print("封存批重算（data/，精确有理数）")
    print(f"证据概率 P(E) = {fr(comp['P_E'])}")
    print(f"现状达标概率 P(面积≥{comp['area_min']} | E) = {fr(comp['tail'])}，"
          f"缺口风险 = {fr(1 - comp['tail'])}（门槛 {fr(comp['thr'])}）")
    print("现状覆盖量分布: " + ", ".join(f"{v}:{fr(p)}"
          for v, p in sorted(comp["pmf"].items()) if p > 0))
    print(f"最优期望总费用 = {fr(comp['expected_cost'])}")
    print("策略分支（仅正概率答复）：")
    tree = []
    policy_tree_lines(comp, comp["root"], 0, tree)
    print("\n".join(tree))
    return 0


def main(argv):
    if len(argv) != 2 or argv[1] not in ("run", "verify", "demo"):
        print("usage: python3 analysis.py {run|verify|demo}", file=sys.stderr)
        return 2
    return {"run": cmd_run, "verify": cmd_verify, "demo": cmd_demo}[argv[1]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
