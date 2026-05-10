"""
CFR 自博弈训练 - 掌心战争
起始状态: 双方 HP=3, 能量=2

算法: 单步反事实遗憾 + 终局信号修正的 MC-CFR
克制链: 大招(3) > 单枪(0) > 清零(8) > 三枪(2) > 大招(3)
        反弹(7) > 单枪(0) | 单枪(0) 部分克制 双枪(1)/三枪(2)
清零条件: HP_MATRIX[对手技能][8] == 1 时回血 (对手出三枪/大招等)
空城被动: 0能量受攻击时额外反伤
"""

import numpy as np
import json
import os
from collections import defaultdict

from dqn_train import (
    NUM_SKILLS, MAX_HP, MAX_ENERGY,
    HP_MATRIX, ZERO_TABLE, SKILL_COST, SKILL_NAMES, SKILL_ATK,
)

# ── 训练参数 ──────────────────────────────────────────────
INIT_HP    = 3      # 起始血量
INIT_EN    = 2      # 起始能量
MAX_ROUNDS = 30
GAMMA      = 0.99   # 折现率
BLEND_STEPS = 5     # 终局信号混合窗口

# 克制链（攻方技能, 守方技能, 描述, 攻方最低能量, 守方最低能量）
# 只有双方能量同时满足时，克制关系才真正成立
COUNTER_CHAIN = [
    (3, 0, "大招>单枪",    3, 1),   # 大招需3能，单枪需1能
    (0, 8, "单枪>清零",    1, 2),   # 单枪需1能，清零需2能
    (8, 2, "清零>三枪",    2, 2),   # 清零/三枪均需2能
    (2, 3, "三枪>大招",    2, 3),   # 三枪需2能，大招需3能
    (7, 0, "反弹>单枪",    1, 1),   # 均需1能
    (0, 1, "单枪>双枪",    1, 2),
    (0, 2, "单枪>三枪",    1, 2),
]

# 能量阶段划分
# phase 0: max(pEn,aEn) == 0          → 双方空城，仅0费技能可用
# phase 1: 1 <= max(pEn,aEn) <= 2     → 部分克制链可触发（无大招）
# phase 2: max(pEn,aEn) >= 3          → 完整克制链，大招可用
def energy_phase(pEn, aEn):
    m = max(pEn, aEn)
    if m == 0:   return 0
    if m <= 2:   return 1
    return 2

PHASE_LABELS = ["双空城", "低能(无大招)", "高能(完整克制链)"]


# ── 状态工具 ──────────────────────────────────────────────

def state_id(pHP, pEn, aHP, aEn):
    return (pHP * 7 + pEn) * 35 + (aHP * 7 + aEn)


def legal_actions(en):
    acts = [s for s in range(NUM_SKILLS) if SKILL_COST[s] <= en]
    return acts if acts else [6]


def game_step(pHP, pEn, aHP, aEn, pSk, aSk):
    """单步结算，返回 (pHP', pEn', aHP', aEn', done, winner)"""
    pEn2 = min(MAX_ENERGY, max(0, pEn - int(SKILL_COST[pSk])) + (1 if pSk == 6 else 0))
    aEn2 = min(MAX_ENERGY, max(0, aEn - int(SKILL_COST[aSk])) + (1 if aSk == 6 else 0))

    if ZERO_TABLE[pSk][aSk]: pEn2 = aEn2 = 0
    if ZERO_TABLE[aSk][pSk]: pEn2 = aEn2 = 0

    aHP2 = max(0, min(MAX_HP, aHP + int(HP_MATRIX[pSk][aSk])))
    pHP2 = max(0, min(MAX_HP, pHP + int(HP_MATRIX[aSk][pSk])))

    if pHP2 <= 0 or aHP2 <= 0:
        return pHP2, pEn2, aHP2, aEn2, True, _winner(pHP2, aHP2)

    # 空城被动
    pAtk = max(int(SKILL_ATK[pSk]), 0)
    aAtk = max(int(SKILL_ATK[aSk]), 0)
    if pHP2 > 0 and pEn2 == 0 and aAtk > 0:
        extra = 1 if int(HP_MATRIX[aSk][pSk]) >= 0 else 0
        aHP2 = max(0, aHP2 - 1 - extra)
    if aHP2 > 0 and aEn2 == 0 and pAtk > 0:
        extra = 1 if int(HP_MATRIX[pSk][aSk]) >= 0 else 0
        pHP2 = max(0, pHP2 - 1 - extra)

    done = pHP2 <= 0 or aHP2 <= 0
    return pHP2, pEn2, aHP2, aEn2, done, _winner(pHP2, aHP2) if done else 0


def _winner(pHP, aHP):
    if pHP <= 0 and aHP <= 0: return 3
    if aHP <= 0: return 1
    return 2


def terminal_value(winner, pHP, aHP):
    """P1 视角终局价值 ∈ [-1.25, 1.25]"""
    if winner == 1: return 1.0 + pHP * 0.05
    if winner == 2: return -1.0 - aHP * 0.05
    return 0.0


def heuristic_value(pHP, pEn, aHP, aEn):
    """
    能量阶段感知的状态评估（P1 视角）

    Phase 0 (空城): 能量为 0 的空城被动主导，HP 优势权重极高
    Phase 1 (低能): 只有部分克制链可用，能量积累价值高
    Phase 2 (高能): 完整克制链激活，相对能量优势决定战略主动权
    """
    if pHP <= 0: return -1.0
    if aHP <= 0: return  1.0

    phase = energy_phase(pEn, aEn)
    hp_adv = (pHP - aHP) / MAX_HP

    if phase == 0:
        # 双方空城：空城被动主导，血量几乎决定一切，能量差无意义
        return hp_adv * 0.95 + ((pEn - aEn) / (MAX_ENERGY + 1)) * 0.05

    elif phase == 1:
        # 低能阶段：充能到 3 才能解锁完整克制链
        # 能量优势 = 能更快到达高能阶段，有战略价值
        en_adv = (pEn - aEn) / (MAX_ENERGY + 1)
        # P1 能量是否已够用 大招(3)
        p1_ulti_ready = 0.1 if pEn >= 3 else 0.0
        p2_ulti_ready = 0.1 if aEn >= 3 else 0.0
        return hp_adv * 0.65 + en_adv * 0.25 + p1_ulti_ready - p2_ulti_ready

    else:
        # 高能阶段：克制链主导，相对能量决定可选克制技能集合
        en_adv = (pEn - aEn) / (MAX_ENERGY + 1)
        # 能使用大招的一方有优势（3能门槛）
        ulti_edge = (0.08 if pEn >= 3 else 0.0) - (0.08 if aEn >= 3 else 0.0)
        # 空城威胁：高能阶段仍需警惕对方打空城
        empty_risk = -0.1 if pEn == 0 and aEn > 0 else 0.0
        return hp_adv * 0.55 + en_adv * 0.30 + ulti_edge + empty_risk


def step_cf_value(php, pen, ahp, aen, alt_p, fixed_a, final_v, gamma):
    """
    P1 在当前状态出 alt_p, P2 出 fixed_a 的反事实价值估算。
    用 (1-gamma) * 即时启发 + gamma * 终局信号 混合。
    """
    nhp, nen, nahp, naen, done, w = game_step(php, pen, ahp, aen, alt_p, fixed_a)
    if done:
        return terminal_value(w, nhp, nahp)
    imm = heuristic_value(nhp, nen, nahp, naen)
    return (1 - gamma) * imm + gamma * final_v


# ── CFR 求解器 ────────────────────────────────────────────

N_STATES = (MAX_HP + 1) * (MAX_ENERGY + 1) * (MAX_HP + 1) * (MAX_ENERGY + 1)  # 1225


class CFRSolver:
    """
    双方独立 CFR，每个玩家维护:
      R[state][action] - 累积反事实遗憾
      S[state][action] - 累积策略权重（平均后趋向 Nash）
    """

    def __init__(self):
        self.R = [np.zeros((N_STATES, NUM_SKILLS), dtype=np.float64) for _ in range(2)]
        self.S = [np.zeros((N_STATES, NUM_SKILLS), dtype=np.float64) for _ in range(2)]
        self.counter_hits  = defaultdict(int)
        self.phase_steps   = [0, 0, 0]   # phase 0/1/2 各自的步数统计
        self.total_steps   = 0
        self.episode_count = 0

    def current_strategy(self, player, sid, legal):
        """遗憾匹配 → 当前混合策略"""
        r = np.maximum(self.R[player][sid], 0)
        mask = np.zeros(NUM_SKILLS)
        mask[legal] = 1.0
        r = r * mask
        total = r.sum()
        if total > 1e-12:
            return r / total
        return mask / mask.sum()

    def average_strategy(self, player, sid, legal):
        """平均策略（Nash 均衡近似）"""
        s = self.S[player][sid] * np.array(
            [1.0 if a in set(legal) else 0.0 for a in range(NUM_SKILLS)]
        )
        total = s.sum()
        if total > 1e-12:
            return s / total
        mask = np.zeros(NUM_SKILLS)
        mask[legal] = 1.0 / len(legal)
        return mask

    def run_episode(self, init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
        """运行完整一局，收集轨迹并执行 CFR 遗憾更新"""
        pHP, pEn, aHP, aEn = init
        traj = []
        done = False; winner = 3; rnd = 0

        while not done and rnd < MAX_ROUNDS:
            rnd += 1
            s  = state_id(pHP, pEn, aHP, aEn)
            l1 = legal_actions(pEn)
            l2 = legal_actions(aEn)
            sg1 = self.current_strategy(0, s, l1)
            sg2 = self.current_strategy(1, s, l2)
            a1  = int(np.random.choice(NUM_SKILLS, p=sg1))
            a2  = int(np.random.choice(NUM_SKILLS, p=sg2))
            traj.append((s, l1, l2, sg1, sg2, a1, a2, pHP, pEn, aHP, aEn))
            self.phase_steps[energy_phase(pEn, aEn)] += 1
            pHP, pEn, aHP, aEn, done, winner = game_step(pHP, pEn, aHP, aEn, a1, a2)

        final_v = terminal_value(winner, pHP, aHP)
        T = len(traj)
        self.episode_count += 1
        self.total_steps   += T

        # ── 反向遍历轨迹，更新 CFR 遗憾 ───────────────────────
        for t, (s, l1, l2, sg1, sg2, a1, a2, php, pen, ahp, aen) in enumerate(traj):
            steps_to_end = T - t - 1
            # gamma 越大 → 越依赖终局信号；越小 → 越依赖即时启发
            g = GAMMA ** steps_to_end if steps_to_end < BLEND_STEPS else 0.0

            # P1 反事实价值向量
            cf1 = np.zeros(NUM_SKILLS)
            for alt in l1:
                cf1[alt] = step_cf_value(php, pen, ahp, aen, alt, a2, final_v, g)
            ev1 = float(sg1 @ cf1)

            # P2 反事实价值向量（零和：取负）
            cf2 = np.zeros(NUM_SKILLS)
            for alt in l2:
                nhp, nen, nahp, naen, d, w = game_step(php, pen, ahp, aen, a1, alt)
                if d:
                    v2 = -terminal_value(w, nhp, nahp)
                else:
                    imm = -heuristic_value(nhp, nen, nahp, naen)
                    v2  = (1 - g) * imm + g * (-final_v)
                cf2[alt] = v2
            ev2 = float(sg2 @ cf2)

            # 更新遗憾
            for alt in l1:
                self.R[0][s][alt] += cf1[alt] - ev1
            for alt in l2:
                self.R[1][s][alt] += cf2[alt] - ev2

            # 更新策略累积（线性加权：越晚的迭代权重越高）
            weight = float(self.episode_count)
            self.S[0][s] += weight * sg1
            self.S[1][s] += weight * sg2

        # ── 克制链统计（仅在双方能量满足条件时才计入）───────────
        for _, _, _, _, _, a1, a2, php, pen, ahp, aen in traj:
            for atk, dfn, lbl, en_atk, en_dfn in COUNTER_CHAIN:
                # P1 是攻方，P2 是守方：检查 P1 能量 >= en_atk 且 P2 能量 >= en_dfn
                if a1 == atk and a2 == dfn and pen >= en_atk and aen >= en_dfn:
                    self.counter_hits[lbl] += 1
                # 反向：P2 是攻方
                elif a2 == atk and a1 == dfn and aen >= en_atk and pen >= en_dfn:
                    self.counter_hits[lbl] += 1

        return winner, T


# ── 评估工具 ──────────────────────────────────────────────

def best_response_winrate(solver, player_br=2, n_eval=300,
                          init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
    """
    player_br 方用针对对手平均策略的贪心最佳响应，
    估算其胜率作为可利用度上界。
    """
    fixed_player = 1 - (player_br - 1)  # 0 or 1
    br_wins = 0

    for _ in range(n_eval):
        pHP, pEn, aHP, aEn = init
        done = False; rnd = 0

        while not done and rnd < MAX_ROUNDS:
            rnd += 1
            s  = state_id(pHP, pEn, aHP, aEn)
            l1 = legal_actions(pEn)
            l2 = legal_actions(aEn)

            # 固定方：用平均策略采样
            fixed_sg = solver.average_strategy(fixed_player, s, l1 if fixed_player == 0 else l2)
            fixed_a  = int(np.random.choice(NUM_SKILLS, p=fixed_sg))

            # BR 方：对固定方当前策略求贪心最佳响应（1步展开）
            br_legal = l2 if player_br == 2 else l1
            best_a = br_legal[0]; best_v = float('-inf')
            for alt in br_legal:
                p1, a = (fixed_a, alt) if fixed_player == 0 else (alt, fixed_a)
                nhp, nen, nahp, naen, d, w = game_step(pHP, pEn, aHP, aEn, p1, a)
                if d:
                    v = terminal_value(w, nhp, nahp) * (1 if player_br == 1 else -1)
                else:
                    v = heuristic_value(nhp, nen, nahp, naen) * (1 if player_br == 1 else -1)
                if v > best_v:
                    best_v = v; best_a = alt

            a1, a2 = (fixed_a, best_a) if fixed_player == 0 else (best_a, fixed_a)
            pHP, pEn, aHP, aEn, done, winner = game_step(pHP, pEn, aHP, aEn, a1, a2)

        if winner == player_br:
            br_wins += 1

    return br_wins / n_eval


# ── 策略分析 ──────────────────────────────────────────────

def print_strategy(solver, pHP, pEn, aHP, aEn, label=""):
    sid = state_id(pHP, pEn, aHP, aEn)
    l1  = legal_actions(pEn)
    l2  = legal_actions(aEn)
    avg1 = solver.average_strategy(0, sid, l1)
    avg2 = solver.average_strategy(1, sid, l2)
    tag  = label or f"HP={pHP} En={pEn} vs HP={aHP} En={aEn}"
    print(f"  [{tag}]")
    p1_top = [(SKILL_NAMES[a], avg1[a]) for a in range(NUM_SKILLS) if avg1[a] > 0.04]
    p2_top = [(SKILL_NAMES[a], avg2[a]) for a in range(NUM_SKILLS) if avg2[a] > 0.04]
    p1_str = "  ".join(f"{n}={p:.0%}" for n, p in sorted(p1_top, key=lambda x: -x[1]))
    p2_str = "  ".join(f"{n}={p:.0%}" for n, p in sorted(p2_top, key=lambda x: -x[1]))
    print(f"    P1: {p1_str}")
    print(f"    P2: {p2_str}")


def analyze_nash(solver, init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
    print("\n[Nash 均衡混合策略分析 — 按能量阶段分层]")

    # ── Phase 0: 空城状态 ─────────────────────────────────
    print(f"\n  === Phase 0: 双方空城（0费技能支配）===")
    for hp in [3, 2, 1]:
        print_strategy(solver, hp, 0, hp, 0, f"对称 HP={hp} En=0")

    # ── Phase 1: 低能（1-2能，大招不可用）────────────────
    print(f"\n  === Phase 1: 低能区（克制链部分激活，无大招）===")
    phase1_states = [
        (*init,           "起始状态 3HP 2能(对称)"),
        (2, 2, 2, 2,      "均势 2HP 2能"),
        (2, 1, 2, 2,      "P1 1能 vs P2 2能"),
        (2, 2, 2, 1,      "P1 2能 vs P2 1能"),
        (3, 2, 3, 1,      "P1能量优势"),
    ]
    for state in phase1_states:
        print_strategy(solver, *state[:4], state[4])

    # ── Phase 2: 高能（3+能，完整克制链）─────────────────
    print(f"\n  === Phase 2: 高能区（完整克制链激活）===")
    phase2_states = [
        (3, 3, 3, 3,      "高能对称 3能"),
        (2, 3, 2, 2,      "P1 大招就绪 vs P2 2能"),
        (2, 2, 2, 3,      "P2 大招就绪 vs P1 2能"),
        (1, 3, 2, 3,      "P1濒死双方高能"),
        (3, 3, 3, 1,      "P2低能 P1大招压制"),
    ]
    for state in phase2_states:
        print_strategy(solver, *state[:4], state[4])

    # ── 起始状态详细 ──────────────────────────────────────
    print(f"\n  === 关键混合状态 ===")
    mixed_states = [
        (init[0], 0, init[2], 0,   "双方空城"),
        (2, 0, 2, 2,               "P1空城 P2有能量"),
        (1, 3, 3, 1,               "P1濒死高能 vs P2健康低能"),
    ]
    for state in mixed_states:
        print_strategy(solver, *state[:4], state[4])

    # ── 克制链能量条件统计 ──────────────────────────────
    total = solver.total_steps
    print(f"\n[克制链触发统计 - 仅统计双方能量条件满足的回合] (共 {total} 步)")
    for _, _, lbl, en_atk, en_dfn in COUNTER_CHAIN:
        cnt = solver.counter_hits.get(lbl, 0)
        rate = cnt / max(total, 1)
        bar  = "█" * int(rate * 50)
        print(f"  {lbl:14s}(攻≥{en_atk}能,守≥{en_dfn}能): "
              f"{cnt:5d}次 {rate:.3%} {bar}")


def save_strategy(solver, path="output/cfr_strategy.npz"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, R0=solver.R[0], R1=solver.R[1],
             S0=solver.S[0], S1=solver.S[1])
    print(f"[保存] 策略 → {path}")


def load_strategy(solver, path="output/cfr_strategy.npz"):
    if not os.path.exists(path):
        return False
    d = np.load(path)
    solver.R[0][:] = d["R0"]; solver.R[1][:] = d["R1"]
    solver.S[0][:] = d["S0"]; solver.S[1][:] = d["S1"]
    print(f"[加载] 策略 ← {path}")
    return True


# ── 主训练循环 ────────────────────────────────────────────

def train_cfr(
    n_iters      = 60_000,
    log_every    = 3_000,
    eval_every   = 10_000,
    save_every   = 20_000,
    init_state   = (INIT_HP, INIT_EN, INIT_HP, INIT_EN),
    save_path    = "output/cfr_strategy.npz",
    resume       = True,
):
    solver = CFRSolver()
    if resume:
        load_strategy(solver, save_path)

    print(f"\n{'='*58}")
    print(f"  CFR 自博弈训练 - 掌心战争")
    print(f"  起始: HP={init_state[0]} 能量={init_state[1]}  (双方对称)")
    print(f"  克制链: 大招>单枪>清零>三枪>大招  |  反弹>单枪")
    print(f"  清零条件: HP_MATRIX[对手技能][8]==1 (回血触发)")
    print(f"  算法: MC-CFR + 终局信号修正 + 线性加权平均策略")
    print(f"  迭代: {n_iters}  log_every: {log_every}")
    print(f"{'='*58}\n")

    wins = [0, 0, 0]  # P1, P2, draw
    log_data = []

    for it in range(1, n_iters + 1):
        winner, rnd = solver.run_episode(init_state)
        wins[winner - 1] += 1

        if it % log_every == 0:
            tot = sum(wins)
            p1_wr = wins[0] / tot
            p2_wr = wins[1] / tot
            draw  = wins[2] / tot
            print(f"[{it:7d}] P1={p1_wr:.1%}  P2={p2_wr:.1%}  平={draw:.1%}  "
                  f"eps_count={solver.episode_count}")

            # 能量阶段分布
            ps = solver.phase_steps
            ps_tot = max(sum(ps), 1)
            print(f"         阶段分布: "
                  f"空城={ps[0]/ps_tot:.0%}  "
                  f"低能={ps[1]/ps_tot:.0%}  "
                  f"高能(克制链)={ps[2]/ps_tot:.0%}")

            top5_counters = sorted(solver.counter_hits.items(), key=lambda x: -x[1])[:3]
            if top5_counters:
                ctr_str = "  ".join(f"{lbl}:{cnt}" for lbl, cnt in top5_counters)
                print(f"         克制触发(能量满足): {ctr_str}")

            print_strategy(solver, *init_state, "起始状态")
            log_data.append({"iter": it, "p1_wr": p1_wr, "p2_wr": p2_wr, "draw": draw})
            wins = [0, 0, 0]

        if it % eval_every == 0:
            br2 = best_response_winrate(solver, player_br=2, n_eval=300, init=init_state)
            br1 = best_response_winrate(solver, player_br=1, n_eval=300, init=init_state)
            exploit = (br1 + br2) / 2
            print(f"  [可利用度估算] BR_P1={br1:.1%}  BR_P2={br2:.1%}  "
                  f"Avg_exploit={exploit:.1%}")

        if it % save_every == 0:
            save_strategy(solver, save_path)

    save_strategy(solver, save_path)

    # 保存训练日志
    log_path = save_path.replace(".npz", "_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"[保存] 训练日志 → {log_path}")

    analyze_nash(solver, init_state)
    return solver


# ── 离线分析入口（加载已有策略）────────────────────────────

def analyze_saved(path="output/cfr_strategy.npz",
                  init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN)):
    solver = CFRSolver()
    if not load_strategy(solver, path):
        print(f"找不到策略文件: {path}")
        return
    analyze_nash(solver, init)
    br2 = best_response_winrate(solver, player_br=2, n_eval=500, init=init)
    br1 = best_response_winrate(solver, player_br=1, n_eval=500, init=init)
    print(f"\n[可利用度] BR_P1={br1:.1%}  BR_P2={br2:.1%}  "
          f"理论Nash差距≈{(br1+br2)/2:.1%}")


# ── DQN 热启动：从 CFR 平均策略采集数据 ───────────────────

def generate_cfr_dataset(solver, n_games=5000,
                          init=(INIT_HP, INIT_EN, INIT_HP, INIT_EN),
                          save_path="output/cfr_dataset.json"):
    """
    用 CFR 平均策略跑 n_games 局，生成 (state, action, outcome) 数据集。
    可用于预训练 DQNAgent（热启动），跳过随机探索阶段。
    """
    dataset = []
    wins = [0, 0, 0]

    for _ in range(n_games):
        pHP, pEn, aHP, aEn = init
        done = False; rnd = 0; traj = []

        while not done and rnd < MAX_ROUNDS:
            rnd += 1
            s  = state_id(pHP, pEn, aHP, aEn)
            l1 = legal_actions(pEn)
            l2 = legal_actions(aEn)
            sg1 = solver.average_strategy(0, s, l1)
            sg2 = solver.average_strategy(1, s, l2)
            a1  = int(np.random.choice(NUM_SKILLS, p=sg1))
            a2  = int(np.random.choice(NUM_SKILLS, p=sg2))
            traj.append((pHP, pEn, aHP, aEn, a1, a2))
            pHP, pEn, aHP, aEn, done, winner = game_step(pHP, pEn, aHP, aEn, a1, a2)

        wins[winner - 1] += 1
        final_v = terminal_value(winner, pHP, aHP)

        for phpt, pent, ahpt, aent, a1t, a2t in traj:
            dataset.append({
                "pHP": phpt, "pEn": pent, "aHP": ahpt, "aEn": aent,
                "a1": a1t, "a2": a2t,
                "winner": winner, "final_value": final_v,
            })

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    tot = sum(wins)
    print(f"[CFR数据集] {len(dataset)} 条  P1={wins[0]/tot:.1%}  P2={wins[1]/tot:.1%}")
    print(f"[保存] → {save_path}")
    return dataset


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_saved()
    elif len(sys.argv) > 1 and sys.argv[1] == "dataset":
        solver = CFRSolver()
        load_strategy(solver)
        generate_cfr_dataset(solver)
    else:
        train_cfr()
