"""
DQN 强化学习训练 - 掌心战争
核心改进：
  1. 收益导向奖励函数：显式奖励 "能量效率"（每伤害点的能量消耗）
  2. 状态归一化向量（12维）：替代 one-hot，支持泛化
  3. 空城Combo奖励：0能出枪 = +0.5 额外奖励
  4. 清零命中奖励：能量清空 + 伤害组合
  5. 技能浪费惩罚：后期有能量不出枪
  6. 优先级经验池（PER）：高价值转移优先回放
  7. 更高 gamma(0.98)：更远期收益
  8. 更保守的 epsilon 衰减：防止过早收敛
"""

import numpy as np
import random
from collections import deque, defaultdict
import json
from datetime import datetime
import copy
import os
import math

# -------------------- 游戏常量 --------------------
NUM_SKILLS = 9
MAX_HP     = 4
MAX_ENERGY = 6

SKILL_ATK = np.array([1, 1, 2, 3, -1, -1, -1, -1, -1], dtype=np.int32)

HP_MATRIX = np.array([
    [  0,  -1,  -1,   0,   0,   0,  -1,   0,  -2],
    [  0,   0,   0,  -2,   0,   0,  -2,   0,  -3],
    [  0,  -1,   0,  -2,  -1,   0,  -3,  -1,   1],
    [ -3,   0,   0,   0,  -3,   0,  -3,  -3,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
    [ -1,  -2,   0,   0,   0,   0,   0,   0,   1],
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],
], dtype=np.int32)

ZERO_TABLE = np.array([
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1],
], dtype=np.int32)

SKILL_NAMES = ['单枪', '双枪', '三枪', '大招', '小防', '大防', '能量', '反弹', '清零']
SKILL_COST  = np.array([1, 2, 2, 3, 0, 1, 0, 1, 2], dtype=np.int32)

# 攻击技能索引（用于识别空城Combo）
ATTACK_SKILLS = {0, 1, 2, 3, 7}  # 单枪、双枪、三枪、大招、反弹（都有正伤害或0伤）
# 有破枪效果的技能（HP_MATRIX[sk][sk] != 0，对应你描述的"双方都是2费时"）
BROKEN_GUN_SKILLS = {1}  # 双枪：HP_MATRIX[1][1] = 0（平局，不亏）


# ==================== 状态编码 ====================
# 用12维归一化向量替代 one-hot，支持泛化
# 维度顺序：[pHP, pEn, aHP, aEn, pHP_norm, pEn_norm, aHP_norm, aEn_norm, pDelta, aDelta, pEfficiency, round_norm]
# 其中 pDelta = pHP - aHP, aDelta = aHP - pHP（对手视角）
N_FEATURES = 22   # 原12维，扩展后22维（含未来收益/对手建模/状态转移）

# ── 能量收益公式常数 ───────────────────────────────
INF = 1e9          # 无穷大
NEG_INF = -1e9     # 无穷小
HP_POSITIVE_WEIGHT = 2.5  # 对方/我方血量变化为正时的额外权重


def _energy_efficiency_reward(my_dmg, their_dmg, a_loss, p_cost):
    """
    能量收益公式（用户指定版本）：

    收益 = ( -Δ对方HP×W_pos + 对方耗能 - (-Δ我方HP)×W_pos ) / 我方能量

    其中：
    - ΔHP < 0 表示掉血（负数），代入公式时 -ΔHP > 0
    - W_pos = 2.5 当 ΔHP > 0（对方/我方回血时）
    - 分母=0 时：分子>0 → +∞，分子<0 → -∞
    """
    if p_cost == 0:
        return 0.0  # 0费技能走情境加成，不走此公式

    # 我方造成伤害：aHP0 - aHP1 > 0 表示打掉对方血
    # 公式中是 -Δ对方HP，其中 Δ = aHP1 - aHP0
    # 所以 -Δ对方HP = aHP0 - aHP1 = my_dmg
    # 但如果 my_dmg < 0（对方回血），则 my_dmg = 0（max处理）

    delta_aHP = -(my_dmg)   # 对方HP变化量（负数=掉血）
    delta_pHP = -(their_dmg)  # 我方HP变化量（负数=掉血）

    if my_dmg > 0:
        delta_aHP *= HP_POSITIVE_WEIGHT
    if their_dmg > 0:
        delta_pHP *= HP_POSITIVE_WEIGHT

    numerator = (-delta_aHP) + a_loss - (-delta_pHP)
    # 即: my_dmg * W + a_loss + their_dmg * W

    if my_dmg > 0:
        delta_aHP_real = my_dmg * HP_POSITIVE_WEIGHT
    else:
        delta_aHP_real = my_dmg  # 回血用负值

    if their_dmg > 0:
        delta_pHP_real = their_dmg * HP_POSITIVE_WEIGHT
    else:
        delta_pHP_real = their_dmg

    numerator = (-delta_aHP_real) + a_loss - (-delta_pHP_real)

    if p_cost == 0:
        if numerator > 0:
            return INF
        elif numerator < 0:
            return NEG_INF
        else:
            return 0.0

    return numerator / p_cost


def state_to_features(pHP, pEn, aHP, aEn, round_num=1, max_round=30):
    """
    将游戏状态转换为归一化特征向量。
    包含绝对值（生命/能量比例）、相对值（HP差、能量差）、效率指标。

    扩展至 22 维特征，增加：
    - 未来收益特征（下回合可用能量、高费技能可达性）
    - 对手建模特征（对手能量效率、攻击倾向）
    - 状态转移价值特征（状态质量分）
    """
    # 绝对值归一化
    pHP_n   = pHP / MAX_HP
    pEn_n   = pEn / MAX_ENERGY
    aHP_n   = aHP / MAX_HP
    aEn_n   = aEn / MAX_ENERGY

    # 相对值（-1 到 1）
    hp_diff   = (pHP - aHP) / MAX_HP        # 我方HP优势
    en_diff   = (pEn - aEn) / MAX_ENERGY    # 我方能量优势

    # 效率指标（我方）
    p_can_attack = 1.0 if pEn >= 1 else 0.0
    p_can_double = 1.0 if pEn >= 2 else 0.0
    p_can_triple = 1.0 if pEn >= 2 else 0.0
    p_can_ulti   = 1.0 if pEn >= 3 else 0.0

    # 回合进度（开局更看重充能，末期更看重决胜）
    round_n  = round_num / max_round
    round_prog = round_num / max_round  # 回合进度 0→1

    # ── 未来收益特征 ─────────────────────────────────────────
    # 下回合能量 = max(0, min(6, curEn - cost + 1(充能)))
    # 充能后我方可达的最大能量
    p_next_en_0  = min(MAX_ENERGY, max(0, pEn) + 1)    # 出0费技能+充能
    p_next_en_charge = min(MAX_ENERGY, max(0, pEn - 0 + 1) + 1)  # 出充能+能量收益
    p_can_next_double = 1.0 if p_next_en_0 >= 2 else 0.0
    p_can_next_ulti   = 1.0 if p_next_en_0 >= 3 else 0.0

    # 下回合能出的最高攻击费
    p_max_atk_cost = max([c for c in SKILL_COST if c <= p_next_en_0 and SKILL_ATK[list(SKILL_COST).index(c)] > 0], default=0)
    p_max_atk_norm = p_max_atk_cost / 3.0  # 归一化到[0,1]

    # 我方当前能量"浪费"程度：能量越多越该出技能
    p_energy_urge = pEn_n * (1.0 - p_can_ulti)  # 有能量但不够出大招 → 充能动机

    # ── 对手建模特征 ────────────────────────────────────────
    # 对手下回合可达能量
    a_next_en = min(MAX_ENERGY, max(0, aEn) + 1)
    a_can_attack = 1.0 if aEn >= 1 else 0.0
    a_can_ulti   = 1.0 if aEn >= 3 else 0.0

    # 对手能量压力：对方能量高 → 我方防守价值上升
    a_energy_threat = aEn_n

    # 我方HP压力：对方能打出多少伤害
    a_potential_dmg = 0.0
    if aEn >= 3:
        a_potential_dmg = 3.0  # 大招
    elif aEn >= 2:
        a_potential_dmg = 2.0  # 三枪
    elif aEn >= 1:
        a_potential_dmg = 1.0  # 单枪
    a_potential_dmg_n = a_potential_dmg / 3.0

    # ── 状态转移价值特征 ─────────────────────────────────────
    # 我方状态质量分（HP权重高，能量中）
    p_state_score = pHP_n * 0.7 + pEn_n * 0.3
    # 对方状态质量分
    a_state_score = aHP_n * 0.7 + aEn_n * 0.3
    # 我方相对状态优势
    state_advantage = p_state_score - a_state_score

    # 濒死标记（HP<=1）
    p_dying = 1.0 if pHP <= 1 else 0.0
    a_dying = 1.0 if aHP <= 1 else 0.0

    # 对方满能量威胁：对方满能量时我方应优先充能/防守
    a_full_energy_threat = 1.0 if aEn >= 6 else 0.0

    # ── 空城标记 ────────────────────────────────────────────
    p_empty_city = 1.0 if pEn == 0 and pHP > 0 else 0.0

    # 综合向量
    return np.array([
        pHP_n, pEn_n, aHP_n, aEn_n,     # 4维：绝对值
        hp_diff,                         # 1维：HP相对优势
        en_diff,                         # 1维：能量相对优势
        p_can_attack,                    # 1维：我能出攻击吗
        p_can_double,                    # 1维：我能出2费技能吗
        p_can_ulti,                     # 1维：我能出大招吗
        round_n,                         # 1维：回合进度
        p_empty_city,                    # 1维：空城标记
        # ── 新增：未来收益特征 (5维) ────────────────────────
        p_can_next_double,              # 下回合能出双枪吗
        p_can_next_ulti,                # 下回合能出大招吗
        p_max_atk_norm,                 # 下回合最高攻击费（归一化）
        p_energy_urge,                  # 能量紧迫度
        # ── 新增：对手建模特征 (4维) ────────────────────────
        a_can_attack,                   # 对手能攻击吗
        a_can_ulti,                     # 对手能放大招吗
        a_potential_dmg_n,              # 对手潜在伤害
        a_energy_threat,               # 对手能量威胁度
        # ── 新增：状态转移价值特征 (3维) ────────────────────
        state_advantage,                # 相对状态优势
        p_dying,                        # 我方濒死
        a_full_energy_threat,           # 对方满能量威胁
    ], dtype=np.float32)


def state_id_from_features(features):
    """
    将12维特征向量转换为整数ID（用于兼容旧接口）。
    只用前4维做离散化：pHP(0-4) × 7 + pEn(0-6) → 35
                        × 7 + aHP(0-4) → 35
                        × 7 + aEn(0-6) → 245
    但返回 1225 以兼容现有网络输入。
    """
    pHP_n = features[0]
    pEn_n = features[1]
    aHP_n = features[2]
    aEn_n = features[3]

    pHP = int(round(pHP_n * MAX_HP))
    pEn = int(round(pEn_n * MAX_ENERGY))
    aHP = int(round(aHP_n * MAX_HP))
    aEn = int(round(aEn_n * MAX_ENERGY))

    pHP = max(0, min(MAX_HP, pHP))
    pEn = max(0, min(MAX_ENERGY, pEn))
    aHP = max(0, min(MAX_HP, aHP))
    aEn = max(0, min(MAX_ENERGY, aEn))

    return (pHP * 7 + pEn) * 35 + (aHP * 7 + aEn)


# ==================== 优先级经验池 ====================
class PrioritizedReplayBuffer:
    """
    优先级经验池（PER）。
    转移优先级 = |TD_error| + epsilon，高价值转移被更频繁回放。
    适合这个游戏中高价值状态（清零Combo、空城Combo）稀少的场景。
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.buf        = deque(maxlen=capacity)
        self.priorities= deque(maxlen=capacity)
        self.alpha      = alpha
        self.beta       = beta_start
        self.beta_inc   = (1.0 - beta_start) / beta_frames
        self.max_prio   = 1.0

    def push(self, s, a, r, s2, d, td_error=None):
        self.buf.append((s, a, r, s2, d))
        prio = abs(td_error) + 1e-5 if td_error is not None else self.max_prio
        self.priorities.append(prio)
        self.max_prio = max(self.max_prio, prio)

    def sample(self, n, batch_idx=None):
        if len(self.buf) < n:
            return None

        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs /= probs.sum()

        if batch_idx is None:
            batch_idx = np.random.choice(len(self.buf), size=n, replace=False, p=probs)

        # IS权重（校正优先级采样带来的偏差）
        weights = (len(self.buf) * probs[batch_idx]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buf[i] for i in batch_idx]
        s_list, a_list, r_list, s2_list, d_list = zip(*batch)
        return (s_list, a_list, r_list, s2_list, d_list,
                weights.tolist(), batch_idx.tolist())

    def update_priorities(self, batch_idx, td_errors):
        for idx, err in zip(batch_idx, td_errors):
            prio = abs(err) + 1e-5
            self.priorities[idx] = prio
            self.max_prio = max(self.max_prio, prio)

    def decay_beta(self):
        self.beta = min(1.0, self.beta + self.beta_inc)

    def __len__(self):
        return len(self.buf)


# ==================== 游戏环境 ====================
class HandGameEnv:
    n_states  = N_FEATURES  # 12 维特征向量
    n_actions = NUM_SKILLS

    def __init__(self, opponent='random', player_passive='none',
                 ai_passive='none', opponent_agent=None):
        self.opponent       = opponent
        self.opponent_agent = opponent_agent
        self.pPassive_type  = player_passive
        self.aPassive_type  = ai_passive
        self.round = 0
        self._reset_state()
        self.a_wait_energy = 0
        self.a_lock_energy = 0

        # ── 对手建模：历史行为统计 ────────────────────────────
        # 记录对手在各种能量下的技能选择概率分布
        self.opp_action_hist = {}  # { (oppEn): {skill: count} }

    def _reset_state(self):
        self.pHP = MAX_HP; self.pEn = 0
        self.aHP = MAX_HP; self.aEn = 0
        self.round   = 1
        self.over    = False
        self.winner  = 0
        self.pSkill  = -1
        self.aSkill  = -1
        self.history = []

    def _state_id(self):
        return (self.pHP * 7 + self.pEn) * 35 + (self.aHP * 7 + self.aEn)

    def _get_state_features(self):
        return state_to_features(self.pHP, self.pEn, self.aHP, self.aEn, self.round)

    def _get_available_actions(self, cur_energy, is_player=True):
        available = [sk for sk in range(NUM_SKILLS) if SKILL_COST[sk] <= cur_energy]
        return available if available else [6]

    def get_player_available_actions(self):
        return self._get_available_actions(self.pEn, is_player=True)

    def _opponent_action(self, pEn0, aHP):
        available = self._get_available_actions(self.aEn, is_player=False)

        if self.aEn == 0 and pEn0 == 0 and aHP > 1 and self.pHP > 1:
            if self.a_lock_energy > 0:
                self.a_lock_energy -= 1
                if 6 in available:
                    return 6
            elif 6 in available and self.a_wait_energy == 0:
                self.a_wait_energy = 1
                self.a_lock_energy = 1
                return 6
        else:
            self.a_wait_energy = 0
            self.a_lock_energy = 0

        if self.opponent == 'random':
            return random.choice(available)
        if self.opponent == 'fixed':
            return available[0] if 0 in available else available[0]
        if self.opponent == 'greedy':
            best, best_dmg = available[0], +999
            psk = self.pSkill if self.pSkill >= 0 else 0
            for a in available:
                dmg = int(HP_MATRIX[a][psk])
                if dmg < best_dmg:
                    best_dmg = dmg
                    best = a
            return best
        if self.opponent == 'snapshot':
            if self.opponent_agent is not None:
                sid = self._state_id()
                return self.opponent_agent.act(sid, train=False, available_actions=available)
            return random.choice(available)
        if self.opponent == 'greedy_model':
            # ── 对手建模：基于历史统计选择最高收益技能 ─────────
            # 利用累积的 `opp_action_hist` 推断对手在当前能量下的行为模式
            key = tuple(sorted([(sk, self.aEn) for sk in available]))
            # 回溯历史：对手在相同/相近能量下最常出的技能
            en_bucket = self.aEn  # 精确能量
            if en_bucket in self.opp_action_hist:
                hist = self.opp_action_hist[en_bucket]
                total = sum(hist.values())
                # 对方最可能出的技能
                likely_skill = max(hist, key=hist.get)
                likely_prob = hist[likely_skill] / total
            else:
                likely_skill = available[0]
                likely_prob = 0.0

            # 计算对方各技能的"对我方伤害+对我方能量影响"
            psk = self.pSkill if self.pSkill >= 0 else 0
            best_score, best_act = float('-inf'), available[0]
            for a in available:
                # 对方技能对我方HP的影响
                p_dmg = max(0, int(HP_MATRIX[a][psk]))
                # 对方技能消耗能量（越低费=对我方能量优势越大）
                cost_penalty = int(SKILL_COST[a])
                # 如果对方在0能且我方有攻击性技能 → 对方危险
                threat = 0
                if self.aEn == 0 and self.pHP > 0:
                    p_atk = max(int(SKILL_ATK[psk]), 0)
                    if p_atk > 0 and int(HP_MATRIX[psk][a]) >= 0:
                        threat = 1.5  # 空城威胁
                # 对方技能得分 = 造成伤害 - 消耗能量 + 空城威胁惩罚
                score = p_dmg * 2.0 - cost_penalty * 0.5 - threat
                if score > best_score:
                    best_score = score
                    best_act = a
            return best_act
        return available[0]

    def _resolve_passive(self, pAtk, aAtk):
        if (self.pPassive_type == 'empty_city'
                and self.pHP > 0 and self.pEn == 0 and aAtk > 0):
            self.aHP -= 1
            pm = (int(HP_MATRIX[self.aSkill][self.pSkill])
                  if 0 <= self.aSkill < 9 and 0 <= self.pSkill < 9 else 0)
            if pm >= 0:
                self.aHP -= 1
            self.aHP = max(self.aHP, 0)

        if (self.aPassive_type == 'empty_city'
                and self.aHP > 0 and self.aEn == 0 and pAtk > 0):
            self.pHP -= 1
            am = (int(HP_MATRIX[self.pSkill][self.aSkill])
                  if 0 <= self.pSkill < 9 and 0 <= self.aSkill < 9 else 0)
            if am >= 0:
                self.pHP -= 1
            self.pHP = max(self.pHP, 0)

    def _check_over(self):
        if self.pHP <= 0 and self.aHP <= 0:
            self.over = True;  self.winner = 3
        elif self.pHP <= 0:
            self.over = True;  self.winner = 2
        elif self.aHP <= 0:
            self.over = True;  self.winner = 1
        else:
            self.over = False; self.winner = 0

    def reset(self):
        self._reset_state()
        self.a_wait_energy = 0
        self.a_lock_energy = 0
        return self._get_state_features()

    def step(self, p_skill, forced_a_skill=None):
        available_p = self._get_available_actions(self.pEn, is_player=True)
        if p_skill not in available_p:
            p_skill = available_p[0]

        pHP0 = self.pHP; aHP0 = self.aHP
        pEn0 = self.pEn; aEn0 = self.aEn

        self.pSkill = p_skill
        if forced_a_skill is not None:
            a_skill = forced_a_skill
        else:
            a_skill = self._opponent_action(pEn0, self.aHP)
        self.aSkill = a_skill

        # ── 对手建模：记录对手技能选择（用于推断对手行为模式）─────
        en_bucket = aEn0  # 用行动前的能量做key
        if en_bucket not in self.opp_action_hist:
            self.opp_action_hist[en_bucket] = {}
        self.opp_action_hist[en_bucket][a_skill] = \
            self.opp_action_hist[en_bucket].get(a_skill, 0) + 1

        # 能量结算
        self.pEn -= int(SKILL_COST[p_skill]); self.aEn -= int(SKILL_COST[a_skill])
        self.pEn = max(self.pEn, 0); self.aEn = max(self.aEn, 0)
        if p_skill == 6: self.pEn += 1
        if a_skill == 6: self.aEn += 1
        self.pEn = min(self.pEn, MAX_ENERGY); self.aEn = min(self.aEn, MAX_ENERGY)

        # 清零
        p_zero = int(ZERO_TABLE[p_skill][a_skill])
        a_zero = int(ZERO_TABLE[a_skill][p_skill])
        if p_zero: self.aEn = 0; self.pEn = 0
        if a_zero: self.pEn = 0; self.aEn = 0

        # HP_MATRIX
        ai_dmg = int(HP_MATRIX[p_skill][a_skill])
        p_dmg  = int(HP_MATRIX[a_skill][p_skill])
        self.aHP += ai_dmg; self.pHP += p_dmg
        self.aHP = max(0, min(self.aHP, MAX_HP))
        self.pHP = max(0, min(self.pHP, MAX_HP))
        self.history.append((p_skill, a_skill))

        self._check_over()
        if self.over:
            self.round += 1
            return self._get_state_features(), self._final_reward(self.round), True, self._info()

        # 空城被动
        pAtk = max(int(SKILL_ATK[p_skill]), 0)
        aAtk = max(int(SKILL_ATK[a_skill]), 0)
        self._resolve_passive(pAtk, aAtk)

        pEn1 = self.pEn; aEn1 = self.aEn
        pHP1 = self.pHP; aHP1 = self.aHP

        self._check_over()
        if not self.over and self.round >= 30:
            self.over = True; self.winner = 3

        step_reward = self._calc_reward(pHP0, aHP0, pEn0, aEn0, p_skill, a_skill, p_zero, a_zero, pHP1, aHP1, pEn1, aEn1)
        done = self.over
        self.round += 1

        if done:
            return self._get_state_features(), self._final_reward(self.round), True, self._info()
        return self._get_state_features(), step_reward, False, self._info()

    def step_simultaneous(self, p_skill, a_skill):
        available_p = self._get_available_actions(self.pEn, is_player=True)
        available_a = self._get_available_actions(self.aEn, is_player=False)
        if p_skill not in available_p:
            p_skill = available_p[0]
        if a_skill not in available_a:
            a_skill = available_a[0]

        pHP0 = self.pHP; aHP0 = self.aHP
        pEn0 = self.pEn; aEn0 = self.aEn

        self.pSkill = p_skill
        self.aSkill = a_skill

        # ── 对手建模：记录对手技能选择 ───────────────────────
        en_bucket = aEn0
        if en_bucket not in self.opp_action_hist:
            self.opp_action_hist[en_bucket] = {}
        self.opp_action_hist[en_bucket][a_skill] = \
            self.opp_action_hist[en_bucket].get(a_skill, 0) + 1

        self.pEn -= int(SKILL_COST[p_skill]); self.aEn -= int(SKILL_COST[a_skill])
        self.pEn = max(self.pEn, 0); self.aEn = max(self.aEn, 0)
        if p_skill == 6: self.pEn += 1
        if a_skill == 6: self.aEn += 1
        self.pEn = min(self.pEn, MAX_ENERGY); self.aEn = min(self.aEn, MAX_ENERGY)

        p_zero = int(ZERO_TABLE[p_skill][a_skill])
        a_zero = int(ZERO_TABLE[a_skill][p_skill])
        if p_zero: self.aEn = 0; self.pEn = 0
        if a_zero: self.pEn = 0; self.aEn = 0

        ai_dmg = int(HP_MATRIX[p_skill][a_skill])
        p_dmg  = int(HP_MATRIX[a_skill][p_skill])
        self.aHP += ai_dmg; self.pHP += p_dmg
        self.aHP = max(0, min(self.aHP, MAX_HP))
        self.pHP = max(0, min(self.pHP, MAX_HP))
        self.history.append((p_skill, a_skill))

        self._check_over()
        if self.over:
            self.round += 1
            return self._get_state_features(), self._final_reward(self.round), True, self._info()

        pAtk = max(int(SKILL_ATK[p_skill]), 0)
        aAtk = max(int(SKILL_ATK[a_skill]), 0)
        self._resolve_passive(pAtk, aAtk)

        self._check_over()
        if not self.over and self.round >= 30:
            self.over = True; self.winner = 3

        pEn1 = self.pEn; aEn1 = self.aEn
        pHP1 = self.pHP; aHP1 = self.aHP

        step_reward = self._calc_reward(pHP0, aHP0, pEn0, aEn0, p_skill, a_skill, p_zero, a_zero, pHP1, aHP1, pEn1, aEn1)
        self.round += 1
        done = self.over
        if done:
            return self._get_state_features(), self._final_reward(self.round), True, self._info()
        return self._get_state_features(), step_reward, False, self._info()

    def _calc_reward(self, pHP0, aHP0, pEn0, aEn0, p_skill, a_skill, p_zero, a_zero, pHP1, aHP1, pEn1, aEn1):
        """
        收益导向奖励函数 v5.

        核心公式（用户指定）：
        收益 = ( -Δ对方HP×W_pos + 对方耗能 - (-Δ我方HP)×W_pos ) / 我方能量

        其中：
        - ΔHP = HP1 - HP0（变化量，负数=掉血）
        - W_pos = 2.5 当 ΔHP > 0（回血时乘以2.5倍）
        - 分母=0：分子>0 → +∞，分子<0 → -∞
        """
        reward = 0.0
        p_cost = int(SKILL_COST[p_skill])
        a_cost = int(SKILL_COST[a_skill])

        # ── 1. 伤害计算 ───────────────────────────────────────
        my_dmg    = max(0, aHP0 - aHP1)
        their_dmg = max(0, pHP0 - pHP1)

        # ── 2. 对方损失能量（基础消耗 + 清零额外）──────────────
        a_loss = a_cost
        if p_zero and HP_MATRIX[a_skill][8] == 1:
            a_loss += aEn0

        # ── 3. 新能量收益公式（核心）────────────────────────────
        if p_cost > 0:
            raw_delta_aHP = aHP0 - aHP1
            raw_delta_pHP = pHP0 - pHP1

            if raw_delta_aHP > 0:
                delta_aHP_w = raw_delta_aHP * HP_POSITIVE_WEIGHT
            else:
                delta_aHP_w = raw_delta_aHP

            if raw_delta_pHP > 0:
                delta_pHP_w = raw_delta_pHP * HP_POSITIVE_WEIGHT
            else:
                delta_pHP_w = raw_delta_pHP

            numerator = (-delta_aHP_w) + a_loss - (-delta_pHP_w)
            reward += numerator / p_cost
        else:
            if p_skill == 6:
                energy_saved = 1
                reward += 0.5 * energy_saved
            elif p_skill == 4:
                if their_dmg > 0:
                    reward += 0.5 * their_dmg
                else:
                    reward -= 0.3

        # ── 4. HP变化（辅助权重，保留用于稳定训练）─────────────
        reward += 2.0 * my_dmg - 1.5 * their_dmg

        # ── 5. 清零失败的额外惩罚 ─────────────────────────────
        if p_skill == 8 and HP_MATRIX[a_skill][8] == -2:
            reward -= 2.0

        # ── 6. 空城Combo奖励 ─────────────────────────────────
        empty_triggered = (int(HP_MATRIX[a_skill][p_skill]) == 1)
        if empty_triggered and pEn0 == 0:
            reward += 1.5

        # ── 7. 防守价值（濒死时）──────────────────────────────
        if self.pHP <= 1 and p_skill in (4, 5):
            reward += 1.0

        # ── 8. 未来收益加成：充能=为下回合高费爆发做准备 ───────
        if p_skill == 6:
            next_en = min(MAX_ENERGY, pEn0 + 1)
            if next_en >= 3:
                reward += 0.5
            elif next_en >= 2:
                reward += 0.2

        # ── 9. 能量紧迫度惩罚：高能量不出攻击 ─────────────────
        if p_skill == 6 and pEn0 >= 2:
            reward -= 0.2

        return reward

    def _final_reward(self, final_round=None):
        if final_round is None:
            final_round = self.round
        if self.winner == 1:
            # 胜利：剩余HP越多越好，能量优势也计入
            hp_bonus = (self.pHP - self.aHP) * 10
            return 200 + hp_bonus
        elif self.winner == 2:
            # 失败：尽量减少损失
            return -150 - (final_round * 0.5)
        else:
            # 平局：中性
            return -50 - (final_round * 0.2)

    def _info(self):
        return {'winner': self.winner, 'round': self.round,
                'pHP': self.pHP, 'pEn': self.pEn,
                'aHP': self.aHP, 'aEn': self.aEn}


def is_attack_skill_possible(energy):
    """判断某能量是否足够出任意攻击技能（>=1费）"""
    return energy >= 1


# ==================== PyTorch DQN ====================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HAS_TORCH = True
except ImportError:
    raise RuntimeError("需要安装 PyTorch：pip install torch")


class QNetwork(nn.Module):
    """
    Q网络：12维归一化特征输入 → 全连接网络 → Q值输出。
    比 one-hot 更容易泛化相似状态。
    """
    def __init__(self, state_dim=N_FEATURES, action_dim=NUM_SKILLS, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,  hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden,    hidden),  nn.LeakyReLU(0.1),
            nn.Linear(hidden,    hidden//2), nn.LeakyReLU(0.1),
            nn.Linear(hidden//2, action_dim),
        )
        # 权重初始化（Xavier）
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    """标准经验池（向后兼容）"""
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s_list, a_list, r_list, s2_list, d_list = zip(*batch)
        return (np.array(s_list, dtype=np.float32),
                list(a_list),
                list(r_list),
                np.array(s2_list, dtype=np.float32),
                list(d_list))

    def __len__(self):
        return len(self.buf)


def _oh(sid, dim=1225):
    """旧接口：one-hot（保留用于兼容）"""
    v = np.zeros(dim, dtype=np.float32)
    v[sid] = 1.0
    return v


def _feat(sid, dim=N_FEATURES):
    """从整数ID恢复特征向量"""
    pHP = sid // 245
    rem = sid % 245
    pEn = rem // 35
    rem = rem % 35
    aHP = rem // 7
    aEn = rem % 7
    return state_to_features(pHP, pEn, aHP, aEn)


class DQNAgent:
    def __init__(self, lr=5e-4, gamma=0.98,
                 eps_start=1.0, eps_end=0.02, eps_decay=0.995,
                 batch_size=64, memory=150_000, hidden=256,
                 stat_window=300, use_per=True):
        self.gamma     = gamma
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.batch_sz  = batch_size
        self.n_actions = NUM_SKILLS
        self.stat_window = stat_window
        self._recent   = deque(maxlen=stat_window)
        self._phase    = 'warmup'
        self.use_per   = use_per

        self.net = QNetwork(N_FEATURES, NUM_SKILLS, hidden).to(DEVICE)
        self.tgt = QNetwork(N_FEATURES, NUM_SKILLS, hidden).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        self.mem = ReplayMemory(memory)

        # PER buffer（独立存储，共享回放）
        if use_per:
            self.per = PrioritizedReplayBuffer(memory, alpha=0.6, beta_start=0.4, beta_frames=100_000)
        else:
            self.per = None

        self.history = []
        self._step_count = 0

    def _state_to_tensor(self, s):
        """将状态（numpy向量或整数ID）转换为tensor"""
        if isinstance(s, (int, np.integer)):
            s = _feat(int(s))
        return torch.FloatTensor(s).unsqueeze(0).to(DEVICE)

    def act(self, s, train=True, available_actions=None):
        if train and random.random() < self.eps:
            if available_actions is not None:
                return random.choice(available_actions)
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            v = self._state_to_tensor(s)
            q_values = self.net(v).squeeze(0)

            if available_actions is not None:
                mask = torch.full((self.n_actions,), float('-inf'), device=DEVICE)
                for a in available_actions:
                    mask[a] = 0.0
                q_values = q_values + mask

            return int(q_values.argmax(0).item())

    def remember(self, s, a, r, s2, d, td_error=None):
        if isinstance(s, (int, np.integer)):
            s = _feat(int(s))
        if isinstance(s2, (int, np.integer)):
            s2 = _feat(int(s2))
        self.mem.push(s, a, r, s2, d)
        if self.use_per and self.per is not None:
            self.per.push(s, a, r, s2, d, td_error)

    def replay(self, n_steps=4):
        if len(self.mem) < self.batch_sz:
            return 0.0

        td_errors = []

        for _ in range(n_steps):
            if self.use_per and self.per is not None and len(self.per) >= self.batch_sz:
                data = self.per.sample(self.batch_sz)
                if data is not None:
                    (S, A, R, S2, D, weights, batch_idx) = data
                    S_t  = torch.FloatTensor(S).to(DEVICE)
                    A_t  = torch.LongTensor(A).to(DEVICE)
                    R_t  = torch.FloatTensor(R).to(DEVICE)
                    S2_t = torch.FloatTensor(S2).to(DEVICE)
                    D_t  = torch.FloatTensor(D).to(DEVICE)
                    W_t  = torch.FloatTensor(weights).to(DEVICE)

                    q = self.net(S_t).gather(1, A_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        best_action = self.net(S2_t).argmax(1)
                        q_target_max = self.tgt(S2_t).gather(1, best_action.unsqueeze(1)).squeeze(1)
                        tgt = R_t + (1 - D_t) * self.gamma * q_target_max

                    # 加权MSE（PER）
                    td = (q - tgt).detach().cpu().numpy()
                    td_errors.extend(td.tolist())
                    loss = (W_t.unsqueeze(1) * (q - tgt.unsqueeze(1)) ** 2).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                    self.opt.step()
                    continue

            S, A, R, S2, D = self.mem.sample(self.batch_sz)
            S_t  = torch.FloatTensor(S).to(DEVICE)
            A_t  = torch.LongTensor(A).to(DEVICE)
            R_t  = torch.FloatTensor(R).to(DEVICE)
            S2_t = torch.FloatTensor(S2).to(DEVICE)
            D_t  = torch.FloatTensor(D).to(DEVICE)

            q = self.net(S_t).gather(1, A_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                best_action = self.net(S2_t).argmax(1)
                q_target_max = self.tgt(S2_t).gather(1, best_action.unsqueeze(1)).squeeze(1)
                tgt = R_t + (1 - D_t) * self.gamma * q_target_max

            td = (q - tgt).detach().cpu().numpy()
            td_errors.extend(td.tolist())
            loss = F.mse_loss(q, tgt)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.opt.step()

        # 更新PER优先级
        if self.use_per and self.per is not None and td_errors:
            self.per.decay_beta()

        return np.mean([abs(e) for e in td_errors]) if td_errors else 0.0

    def update_target(self):
        self.tgt.load_state_dict(self.net.state_dict())

    def decay(self):
        if not hasattr(self, '_phase') or self._phase == 'warmup':
            return
        draws = sum(1 for r in self._recent if r == 3)
        wins  = sum(1 for r in self._recent if r == 1)
        total = len(self._recent)
        if total == 0:
            return
        win_rate  = wins / total
        draw_rate = draws / total
        # 胜率高且对手在拖时间时暂停衰减
        if win_rate >= 0.20 and draw_rate > 0.40:
            return
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def record(self, winner):
        self._recent.append(winner)

    def clone_net(self):
        snapshot = QNetwork(N_FEATURES, NUM_SKILLS).to(DEVICE)
        snapshot.load_state_dict(copy.deepcopy(self.net.state_dict()))
        snapshot.eval()
        return snapshot


# ==================== 技能胜率追踪器 ====================
class SkillWinRateTracker:
    """
    追踪每个(我HP, 我能量, 敌HP, 敌能量)组合下各技能的胜率。

    记录结构：
        stats[pHP][pEn_bucket][aHP][aEn_bucket][skill] = {'w': w, 'd': d, 'l': l, 'total': n}
    其中 pEn_bucket ∈ {0..4}（能量>=5统一归桶4），aEn_bucket 同理。

    输出到 output/skill_winrate_YYYYMMDD_HHMMSS.json
    """

    def __init__(self, en_threshold=5):
        self.en_threshold = en_threshold
        self.stats = {}
        self.episode_records = []

    def _en_bucket(self, en):
        return min(en, self.en_threshold - 1)

    def _ensure_path(self, pHP, pEn_b, aHP, aEn_b, skill):
        if pHP not in self.stats:
            self.stats[pHP] = {}
        if pEn_b not in self.stats[pHP]:
            self.stats[pHP][pEn_b] = {}
        if aHP not in self.stats[pHP][pEn_b]:
            self.stats[pHP][pEn_b][aHP] = {}
        if aEn_b not in self.stats[pHP][pEn_b][aHP]:
            self.stats[pHP][pEn_b][aHP][aEn_b] = {}
        if skill not in self.stats[pHP][pEn_b][aHP][aEn_b]:
            self.stats[pHP][pEn_b][aHP][aEn_b][skill] = {
                'w': 0, 'd': 0, 'l': 0, 'total': 0
            }

    def start_episode(self):
        self.episode_records = []

    def record_step(self, pHP, pEn, aHP, aEn, skill, outcome):
        pEn_b = self._en_bucket(pEn)
        aEn_b = self._en_bucket(aEn)
        self._ensure_path(pHP, pEn_b, aHP, aEn_b, skill)
        entry = {
            'pHP': pHP, 'pEn_b': pEn_b,
            'aHP': aHP, 'aEn_b': aEn_b,
            'skill': skill, 'outcome': outcome
        }
        self.episode_records.append(entry)

    def end_episode(self, winner):
        if winner == 1:
            outcome = 1
        elif winner == 2:
            outcome = -1
        else:
            outcome = 0

        for rec in self.episode_records:
            hp = rec['pHP']
            pEn_b = rec['pEn_b']
            aHP = rec['aHP']
            aEn_b = rec['aEn_b']
            sk = rec['skill']
            s = self.stats[hp][pEn_b][aHP][aEn_b][sk]
            s['total'] += 1
            if outcome == 1:
                s['w'] += 1
            elif outcome == -1:
                s['l'] += 1
            else:
                s['d'] += 1

    # ── 核心分析函数 ────────────────────────────────────────

    def _win_rate(self, s):
        if s['total'] == 0:
            return None
        return s['w'] / s['total']

    def _parse_rate(self, s):
        """返回 (胜率, 平率, 负率, 样本数)"""
        if s['total'] == 0:
            return None, None, None, 0
        t = s['total']
        return s['w'] / t, s['d'] / t, s['l'] / t, t

    def get_best_skill_per_state(self, min_samples=5):
        """
        返回 {(pHP, pEn, aHP, aEn): {skill: (wr, dr, lr, n)}}
        过滤掉样本数不足的状态。
        """
        results = {}
        for pHP in self.stats:
            for pEn_b in self.stats[pHP]:
                for aHP in self.stats[pHP][pEn_b]:
                    for aEn_b in self.stats[pHP][pEn_b][aHP]:
                        state_key = (pHP, pEn_b, aHP, aEn_b)
                        skills_data = {}
                        has_data = False
                        for sk in range(NUM_SKILLS):
                            if sk in self.stats[pHP][pEn_b][aHP][aEn_b]:
                                s = self.stats[pHP][pEn_b][aHP][aEn_b][sk]
                                if s['total'] >= min_samples:
                                    skills_data[sk] = self._parse_rate(s)
                                    has_data = True
                        if has_data:
                            results[state_key] = skills_data
        return results

    def get_summary_matrix(self, min_samples=3):
        """
        返回聚合矩阵：按(我HP, 我能量桶)分组，忽略对手具体状态。
        用于观察"在某种自身状态下，出哪个技能最好"。
        格式：matrix[pHP][pEn_b] = {skill: (wr, n)}
        """
        agg = [defaultdict(lambda: defaultdict(lambda: {'w': 0, 'total': 0}))
               for _ in range(5)]
        for pHP in self.stats:
            for pEn_b in self.stats[pHP]:
                for aHP in self.stats[pHP][pEn_b]:
                    for aEn_b in self.stats[pHP][pEn_b][aHP]:
                        for sk, s in self.stats[pHP][pEn_b][aHP][aEn_b].items():
                            if s['total'] >= min_samples:
                                agg[pHP][pEn_b][sk]['w'] += s['w']
                                agg[pHP][pEn_b][sk]['total'] += s['total']
        return agg

    def get_relative_matrix(self, min_samples=3):
        """
        返回相对优势矩阵：按(我HP-敌HP, 我能量-敌能量)分组。
        正值表示我方优势，负值表示敌方优势。
        格式：matrix[hp_diff+2][en_diff+2] = {skill: (wr, n)}
        """
        agg = [[defaultdict(lambda: {'w': 0, 'total': 0})
                for _ in range(5)] for _ in range(5)]
        for pHP in self.stats:
            for pEn_b in self.stats[pHP]:
                for aHP in self.stats[pHP][pEn_b]:
                    for aEn_b in self.stats[pHP][pEn_b][aHP]:
                        hp_diff = max(0, min(4, pHP - aHP + 2))
                        en_diff = max(0, min(4, pEn_b - aEn_b + 2))
                        for sk, s in self.stats[pHP][pEn_b][aHP][aEn_b].items():
                            if s['total'] >= min_samples:
                                agg[hp_diff][en_diff][sk]['w'] += s['w']
                                agg[hp_diff][en_diff][sk]['total'] += s['total']
        return agg

    # ── 文件输出 ────────────────────────────────────────────

    def _format_path(self):
        from datetime import datetime as _dt
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join('output', f'skill_winrate_{ts}.json')

    def save_to_file(self, path=None):
        if path is None:
            path = self._format_path()

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # 构建可序列化的字典
        data = {
            'meta': {
                'en_threshold': self.en_threshold,
                'en_bucket_desc': (
                    [f'En={i}' for i in range(self.en_threshold - 1)]
                    + [f'En>={self.en_threshold - 1}']
                ),
                'total_samples': sum(
                    s['total']
                    for pHP in self.stats
                    for pEn_b in self.stats[pHP]
                    for aHP in self.stats[pHP][pEn_b]
                    for aEn_b in self.stats[pHP][pEn_b][aHP]
                    for s in self.stats[pHP][pEn_b][aHP][aEn_b].values()
                ),
            },
            # 详细数据：按(pHP, pEn, aHP, aEn) → {skill: {w,d,l,total}}
            'detail': {},
            # 最佳技能推荐：按(pHP, pEn)聚合
            'own_state_best': {},
            # 相对优势推荐：按(hp_diff, en_diff)聚合
            'relative_best': {},
        }

        # detail
        for pHP in sorted(self.stats.keys()):
            for pEn_b in sorted(self.stats[pHP].keys()):
                for aHP in sorted(self.stats[pHP][pEn_b].keys()):
                    for aEn_b in sorted(self.stats[pHP][pEn_b][aHP].keys()):
                        key = f'{pHP}_{pEn_b}_{aHP}_{aEn_b}'
                        skills_out = {}
                        for sk, s in self.stats[pHP][pEn_b][aHP][aEn_b].items():
                            wr, dr, lr, n = self._parse_rate(s)
                            skills_out[str(sk)] = {
                                'win_rate': round(wr, 4) if wr is not None else None,
                                'draw_rate': round(dr, 4) if dr is not None else None,
                                'loss_rate': round(lr, 4) if lr is not None else None,
                                'samples': n,
                                'skill_name': SKILL_NAMES[sk],
                            }
                        data['detail'][key] = skills_out

        # own_state_best：只看自身状态，不管对手
        summary = self.get_summary_matrix(min_samples=3)
        for pHP in range(5):
            for pEn_b in range(5):
                best_sk, best_wr, best_n = None, -1.0, 0
                row = {}
                for sk in range(NUM_SKILLS):
                    s = summary[pHP][pEn_b][sk]
                    if s['total'] >= 3:
                        wr = s['w'] / s['total']
                        row[str(sk)] = {
                            'win_rate': round(wr, 4),
                            'samples': s['total'],
                            'skill_name': SKILL_NAMES[sk],
                        }
                        if wr > best_wr:
                            best_wr = wr
                            best_sk = sk
                            best_n = s['total']
                if row:
                    en_label = f'En={pEn_b}' if pEn_b < 4 else 'En≥5'
                    data['own_state_best'][f'{pHP}_{pEn_b}'] = {
                        'pHP': pHP, 'pEn_bucket': pEn_b,
                        'pEn_label': en_label,
                        'best_skill': best_sk,
                        'best_skill_name': SKILL_NAMES[best_sk] if best_sk is not None else None,
                        'best_win_rate': round(best_wr, 4) if best_wr > 0 else None,
                        'best_samples': best_n,
                        'all_skills': row,
                    }

        # relative_best：按相对差值分组
        rel = self.get_relative_matrix(min_samples=3)
        for di in range(5):
            for dj in range(5):
                best_sk, best_wr = None, -1.0
                row = {}
                for sk in range(NUM_SKILLS):
                    s = rel[di][dj][sk]
                    if s['total'] >= 3:
                        wr = s['w'] / s['total']
                        row[str(sk)] = {
                            'win_rate': round(wr, 4),
                            'samples': s['total'],
                            'skill_name': SKILL_NAMES[sk],
                        }
                        if wr > best_wr:
                            best_wr = wr
                            best_sk = sk
                if row:
                    hp_diff_val = di - 2
                    en_diff_val = dj - 2
                    data['relative_best'][f'{di}_{dj}'] = {
                        'hp_diff': hp_diff_val,
                        'en_diff': en_diff_val,
                        'description': f'我HP-{abs(hp_diff_val)}' if hp_diff_val < 0 else (
                            f'敌HP-{abs(hp_diff_val)}' if hp_diff_val > 0 else 'HP相等'
                        ),
                        'best_skill': best_sk,
                        'best_skill_name': SKILL_NAMES[best_sk] if best_sk is not None else None,
                        'best_win_rate': round(best_wr, 4) if best_wr > 0 else None,
                        'all_skills': row,
                    }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return path

    def print_summary(self):
        """终端打印（简略版）"""
        total = sum(
            s['total']
            for pHP in self.stats
            for pEn_b in self.stats[pHP]
            for aHP in self.stats[pHP][pEn_b]
            for aEn_b in self.stats[pHP][pEn_b][aHP]
            for s in self.stats[pHP][pEn_b][aHP][aEn_b].values()
        )
        print(f"\n  技能胜率追踪：累计 {total} 条记录")
        print(f"  详细结果已保存 → output/skill_winrate_*.json")


# ==================== 快照对手池 ====================
class SnapshotPool:
    def __init__(self, maxlen=10):
        self.pool   = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def add(self, net_snapshot):
        self.pool.append(net_snapshot)

    def sample(self):
        if not self.pool:
            return None
        weights = np.exp(np.linspace(0, 2, len(self.pool)))
        weights /= weights.sum()
        idx = np.random.choice(len(self.pool), p=weights)
        return list(self.pool)[idx]

    def __len__(self):
        return len(self.pool)


class SnapshotAgent:
    def __init__(self, net):
        self.net = net

    def act(self, sid, train=False, available_actions=None):
        with torch.no_grad():
            v = torch.FloatTensor([_feat(sid)]).to(DEVICE)
            q_values = self.net(v).squeeze(0)
            if available_actions is not None:
                mask = torch.full((NUM_SKILLS,), float('-inf'), device=DEVICE)
                for a in available_actions:
                    mask[a] = 0.0
                q_values = q_values + mask
            return int(q_values.argmax(0).item())


# ==================== 单局训练工具 ====================
def _run_episode(env, agent, train=True):
    s, done = env.reset(), False
    ep_buffer = []
    while not done:
        available = env.get_player_available_actions()
        a = agent.act(s, train=train, available_actions=available)
        s2, r, done, info = env.step(a)
        if train:
            ep_buffer.append((s, a, r, s2, done))
        s = s2
    if train:
        for s, a, r, s2, d in ep_buffer:
            agent.remember(s, a, r, s2, d)
        for _ in range(len(ep_buffer)):
            agent.replay()
    return info


def _run_episode_with_tracker(env, agent, tracker, train=True):
    """
    带胜率追踪的 episode 运行。
    每步记录 (我HP, 我能量, 敌HP, 敌能量, 技能) → 局结束时统一写入 tracker。
    状态向量 s = [pHP_n, pEn_n, aHP_n, aEn_n, ...]（前4维为归一化值）。
    """
    s, done = env.reset(), False
    ep_buffer = []
    tracker.start_episode()

    while not done:
        available = env.get_player_available_actions()
        a = agent.act(s, train=train, available_actions=available)
        s2, r, done, info = env.step(a)

        # 解析行动前的 4 维原始状态
        pHP = max(0, min(MAX_HP, int(round(s[0] * MAX_HP))))
        pEn = max(0, min(MAX_ENERGY, int(round(s[1] * MAX_ENERGY))))
        aHP = max(0, min(MAX_HP, int(round(s[2] * MAX_HP))))
        aEn = max(0, min(MAX_ENERGY, int(round(s[3] * MAX_ENERGY))))

        tracker.record_step(pHP, pEn, aHP, aEn, a, None)

        if train:
            ep_buffer.append((s, a, r, s2, done))
        s = s2

    if train:
        for s, a, r, s2, d in ep_buffer:
            agent.remember(s, a, r, s2, d)
        for _ in range(len(ep_buffer)):
            agent.replay()

    tracker.end_episode(info['winner'])
    return info


def _run_selfplay(env, agent, train=True):
    s = env.reset()
    done = False
    ep_buffer = []
    while not done:
        p_avail = env.get_player_available_actions()
        a_avail = env._get_available_actions(env.aEn, is_player=False)

        p_act = agent.act(s, train=train, available_actions=p_avail)
        a_act = agent.act(s, train=train, available_actions=a_avail)

        s2, r, done, info = env.step_simultaneous(p_act, a_act)

        if train:
            ep_buffer.append((s, p_act, r, s2, done))
        s = s2

    fr_p = env._final_reward()
    fr_a = -fr_p

    if ep_buffer:
        s, p_act, r, s2, _ = ep_buffer[-1]
        ep_buffer[-1] = (s, p_act, fr_p, s2, True)

    if train:
        for s, a, r, s2, d in ep_buffer:
            agent.remember(s, a, r, s2, d)
        for _ in range(len(ep_buffer)):
            agent.replay()

    return {'winner': info['winner'], 'round': info['round'],
            'final_reward_p': fr_p, 'final_reward_a': fr_a}


# ==================== 混合训练主函数 ====================
def train_mixed(
    warmup_eps      = 1000,
    total_eps       = 15000,
    random_ratio    = 0.20,
    greedy_ratio    = 0.10,
    fixed_ratio     = 0.05,
    target_every    = 100,
    snapshot_every  = 500,
    refine_start    = 0.75,
    save            = 'output/dqn_mixed_v2.pt',
    log_every       = 200,
    passive         = 'empty_city',
    lr              = 5e-4,
    gamma           = 0.98,
    hidden          = 256,
    use_per         = True,
    winrate_every   = 1000,
):
    agent = DQNAgent(lr=lr, gamma=gamma, hidden=hidden, use_per=use_per)
    # ── 技能胜率追踪器 ──────────────────────────────────────
    tracker = SkillWinRateTracker(en_threshold=5)

    selfplay_ratio = max(0.0, 1.0 - random_ratio - greedy_ratio - fixed_ratio)
    opp_types = ['selfplay', 'random', 'greedy', 'fixed']
    opp_probs = np.array([selfplay_ratio, random_ratio, greedy_ratio, fixed_ratio])
    opp_probs /= opp_probs.sum()

    refine_from = int(total_eps * refine_start)

    stats = {t: {'w': 0, 'd': 0, 'l': 0} for t in opp_types}
    stats['total'] = {'w': 0, 'd': 0, 'l': 0}

    def _update_stats(info, opp_type):
        w = info['winner']
        if   w == 1: key = 'w'
        elif w == 2: key = 'l'
        else:        key = 'd'
        stats[opp_type][key] += 1
        stats['total'][key]  += 1

    def _print_log(ep, phase):
        t  = stats['total']
        tot = t['w'] + t['d'] + t['l']
        if tot == 0: return
        wr = t['w'] / tot
        print(f"[{phase} E{ep:6d}] WR={wr:.1%} "
              f"({t['w']}W/{t['d']}D/{t['l']}L) eps={agent.eps:.4f} "
              f"mem={len(agent.mem)}")
        for ot in opp_types:
            s = stats[ot]
            n = s['w'] + s['d'] + s['l']
            if n > 0:
                print(f"   vs {ot:8s}: {n}局 "
                      f"W={s['w']} D={s['d']} L={s['l']} "
                      f"WR={s['w']/n:.1%}")
        for ot in opp_types:
            stats[ot] = {'w': 0, 'd': 0, 'l': 0}
        stats['total'] = {'w': 0, 'd': 0, 'l': 0}

    # 阶段 0：预热
    print(f"\n{'='*55}")
    print(f"  阶段0：预热期 ({warmup_eps} 局，纯随机对手)")
    print(f"  奖励函数：收益导向 + 空城Combo + 清零奖励")
    print(f"  状态编码：{N_FEATURES}维归一化特征向量")
    print(f"  PER优先级经验池：{'开启' if use_per else '关闭'}")
    print(f"{'='*55}")
    env_rand = HandGameEnv(opponent='random',
                           player_passive=passive, ai_passive=passive)
    for ep in range(1, warmup_eps + 1):
        info = _run_episode_with_tracker(env_rand, agent, tracker, train=True)
        agent.record(info['winner'])
        _update_stats(info, 'random')
        if ep % target_every == 0:
            agent.update_target()
        if ep % log_every == 0:
            _print_log(ep, 'Warmup')

    # 阶段 1 + 2
    print(f"\n{'='*55}")
    print(f"  阶段1：混合期 → 阶段2：精炼期（合计 {total_eps} 局）")
    print(f"  混合比例 selfplay={selfplay_ratio:.0%} random={random_ratio:.0%} "
          f"greedy={greedy_ratio:.0%} fixed={fixed_ratio:.0%}")
    print(f"  精炼期从第 {refine_from} 局开始")
    print(f"{'='*55}")

    snapshot_pool = SnapshotPool(maxlen=10)
    envs = {
        'random'  : HandGameEnv(opponent='random',
                                player_passive=passive, ai_passive=passive),
        'greedy'  : HandGameEnv(opponent='greedy',
                                player_passive=passive, ai_passive=passive),
        'fixed'   : HandGameEnv(opponent='fixed',
                                player_passive=passive, ai_passive=passive),
    }

    for ep in range(1, total_eps + 1):
        agent._phase = 'train'
        if ep >= refine_from:
            opp_type = 'selfplay'
        else:
            opp_type = np.random.choice(opp_types, p=opp_probs)

        if opp_type == 'selfplay':
            snap = snapshot_pool.sample()
            env_sp = HandGameEnv(opponent='snapshot',
                                 player_passive=passive, ai_passive=passive,
                                 opponent_agent=SnapshotAgent(snap) if snap else None)
            info = _run_selfplay(env_sp, agent, train=True)
        else:
            info = _run_episode_with_tracker(envs[opp_type], agent, tracker, train=True)

        agent.record(info['winner'])
        agent.decay()

        if ep % snapshot_every == 0:
            snapshot_pool.add(agent.clone_net())

        _update_stats(info, opp_type)

        if ep % target_every == 0:
            agent.update_target()

        if ep % log_every == 0:
            phase = '精炼期' if ep >= refine_from else '混合期'
            _print_log(ep, phase)
            agent.history.append({
                'ep': ep, 'eps': agent.eps,
                'phase': phase,
            })

        if ep % winrate_every == 0 and ep > 0:
            best = tracker.get_best_skill_per_state(min_samples=5)
            if best:
                print(f"\n  ── 技能胜率分析 (E{ep}) ──")
                for (pHP, pEn_b, aHP, aEn_b), skills_data in sorted(best.items()):
                    best_sk = max(skills_data, key=lambda sk: skills_data[sk][0] or 0)
                    wr, dr, lr, n = skills_data[best_sk]
                    pEn_label = f"En={pEn_b}" if pEn_b < 4 else "En≥5"
                    aEn_label = f"En={aEn_b}" if aEn_b < 4 else "En≥5"
                    print(f"    我HP={pHP} {pEn_label} vs敌HP={aHP} {aEn_label}: "
                          f"{SKILL_NAMES[best_sk]}(胜{wr:.0%}平{dr:.0%}负{lr:.0%}, n={n})")
            else:
                print(f"\n  ── 技能胜率 (E{ep}): 样本不足 ──")

        if ep % 1000 == 0:
            os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
            torch.save({'policy': agent.net.state_dict(),
                        'hist': agent.history}, save)

    os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
    torch.save({'policy': agent.net.state_dict(),
                'hist': agent.history}, save)

    # ── 保存技能胜率详细统计 ─────────────────────────────────
    winrate_path = tracker.save_to_file()
    print(f"\n[完成] 模型已保存 → {save}")
    print(f"[完成] 胜率分析已保存 → {winrate_path}")
    tracker.print_summary()
    return agent


# ==================== 评估 ====================
def evaluate(agent, games=500, opponent='random'):
    env   = HandGameEnv(opponent=opponent)
    wins  = draws = losses = 0
    sk_plays = np.zeros(NUM_SKILLS, dtype=np.float64)

    for _ in range(games):
        s, done = env.reset(), False
        while not done:
            available = env.get_player_available_actions()
            a = agent.act(s, train=False, available_actions=available)
            sk_plays[a] += 1
            s2, r, done, info = env.step(a)
            s = s2
        w = info['winner']
        if   w == 1: wins   += 1
        elif w == 0: draws  += 1
        else:        losses += 1

    tot = wins + draws + losses
    print(f"\n[评估 vs {opponent}] {games}局 | "
          f"胜={wins}({wins/tot:.1%}) "
          f"平={draws}({draws/tot:.1%}) "
          f"负={losses}({losses/tot:.1%})")
    total_plays = sk_plays.sum()
    for i in range(NUM_SKILLS):
        if sk_plays[i] > 0:
            print(f"  {SKILL_NAMES[i]:4s}: {sk_plays[i]:.0f}次 "
                  f"({sk_plays[i]/total_plays:.1%})")
    return wins / max(tot, 1)


# ==================== 热力图 ====================
def build_table(agent):
    t = np.full((5, 7, 5, 7), -1, dtype=np.int8)
    env_temp = HandGameEnv()
    for pHP in range(5):
        for pEn in range(7):
            for aHP in range(5):
                for aEn in range(7):
                    sid = (pHP * 7 + pEn) * 35 + (aHP * 7 + aEn)
                    env_temp.pHP, env_temp.pEn = pHP, pEn
                    env_temp.aHP, env_temp.aEn = aHP, aEn
                    available = env_temp.get_player_available_actions()
                    t[pHP, pEn, aHP, aEn] = agent.act(sid, train=False, available_actions=available)
    return t


def print_heatmap(table, pEn=0, aEn=0, title=''):
    print(f"\n{title or f'最优技能 (pEn={pEn} aEn={aEn})'}")
    header = 'P\\A'
    print(f"{header:>5s}", end="")
    for aHP in range(5):
        print(f"  aHP{aHP}", end="")
    print()
    for pHP in range(5):
        print(f"  pHP{pHP} ", end="")
        for aHP in range(5):
            s = SKILL_NAMES[table[pHP, pEn, aHP, aEn]]
            print(f"  {s:>4s}", end="")
        print()
    print("图例: " + " | ".join(f"{i}:{n}" for i, n in enumerate(SKILL_NAMES)))


# ==================== 奖励调试工具 ====================
def debug_reward():
    """打印关键场景的理论奖励值，验证奖励函数正确性"""
    print("\n=== 奖励函数调试 ===")
    print("\n场景1: 单枪(1费) vs 对方空城 → 造成1伤")
    env = HandGameEnv()
    env.pHP, env.pEn, env.aHP, env.aEn = 4, 1, 3, 0
    env.round = 5
    r = env._calc_reward(4, 3, 1, 0, 0, 6, 0, 0, 4, 2, 1, 0)
    print(f"  我:单枪 vs 对方:能量 → 我造成1伤，能量归0, r={r:.2f}")
    print(f"  预期: 基础3.0 + 效率(1/1-1)×2=3.0 ≈ 3.0")

    print("\n场景2: 三枪(2费) vs 对方空城 → 造成2伤")
    env = HandGameEnv()
    env.pHP, env.pEn, env.aHP, env.aEn = 4, 2, 3, 0
    env.round = 5
    r = env._calc_reward(4, 3, 2, 0, 2, 6, 0, 0, 4, 1, 2, 0)
    print(f"  我:三枪 vs 对方:能量 → 造成2伤, r={r:.2f}")
    print(f"  预期: 基础6.0 + 效率(2/2-1)×2=6.0 ≈ 6.0")

    print("\n场景3: 三枪(2费,超级空城) vs 对方空城 → 造成2伤+空城被动2伤")
    env = HandGameEnv(player_passive='empty_city', ai_passive='empty_city')
    env.pHP, env.pEn, env.aHP, env.aEn = 4, 0, 3, 0
    env.round = 5
    env.pSkill, env.aSkill = 2, 6
    # 先计算伤害：HP_MATRIX[2][6] = -3，aHP:3→0
    r = env._calc_reward(4, 3, 0, 0, 2, 6, 0, 0, 4, 0, 0, 0)
    print(f"  我:三枪(0能) vs 对方:能量(0能) → 造成2伤+空城被动2伤, r={r:.2f}")
    print(f"  预期: 基础12.0 + 空城Combo1.0 + 效率4.0 ≈ 17.0")

    print("\n场景4: 单枪(1费) vs 对方小防(0费) → 被防住，不掉血")
    env = HandGameEnv()
    env.pHP, env.pEn, env.aHP, env.aEn = 4, 1, 4, 1
    env.round = 5
    r = env._calc_reward(4, 4, 1, 1, 0, 4, 0, 0, 4, 4, 0, 1)
    print(f"  我:单枪 vs 对方:小防 → 不掉血, r={r:.2f}")
    print(f"  预期: 基础0 + 效率0.0 ≈ 0.0")

    print("\n场景5: 清零(2费) vs 对方2能 → 清零命中+造成1伤")
    env = HandGameEnv()
    env.pHP, env.pEn, env.aHP, env.aEn = 4, 2, 4, 2
    env.round = 5
    # 清零[8][0] = 1, HP_MATRIX[8][0] = -2 (造成2伤害)
    r = env._calc_reward(4, 4, 2, 2, 8, 0, 1, 0, 4, 2, 0, 0)
    print(f"  我:清零 vs 对方:单枪 → 清零命中+造成2伤, r={r:.2f}")
    print(f"  预期: 基础6.0 + 清零2.0+伤害加成3.0 + 效率4.0 ≈ 15.0")

    print("\n场景6: 双枪(2费) vs 对方清零(2费) → 被清零，-2伤")
    env = HandGameEnv()
    env.pHP, env.pEn, env.aHP, env.aEn = 4, 2, 4, 2
    env.round = 5
    r = env._calc_reward(4, 4, 2, 2, 1, 8, 0, 1, 2, 4, 0, 0)
    print(f"  我:双枪 vs 对方:清零 → 被清零，-2伤, r={r:.2f}")
    print(f"  预期: 基础-4.0 + 被清零惩罚 ≈ -4.0")

    print("\n场景7: 能量(0费) vs 对方能量(0费) → 双方充能")
    env = HandGameEnv()
    env.pHP, env.pEn, env.aHP, env.aEn = 4, 0, 4, 0
    env.round = 5
    r = env._calc_reward(4, 4, 0, 0, 6, 6, 0, 0, 4, 4, 1, 1)
    print(f"  我:能量 vs 对方:能量 → 双方+1能, r={r:.2f}")
    print(f"  预期: 充能奖励0.3 ≈ 0.3")

    print("\n场景8: 反弹(1费) vs 对方三枪(2费) → 反弹造成1伤，对方-3伤")
    env = HandGameEnv()
    env.pHP, env.pEn, env.aHP, env.aEn = 4, 1, 4, 2
    env.round = 5
    # 反弹 sk=7, 三枪 sk=2
    # HP_MATRIX[7][2] = 0（反弹不造成伤害）, HP_MATRIX[2][7] = -1（对方-1伤）
    # SKILL_ATK[7] = -1（反弹）
    r = env._calc_reward(4, 4, 1, 2, 7, 2, 0, 0, 4, 3, 0, 2)
    print(f"  我:反弹 vs 对方:三枪 → 对方-1伤，我+1能, r={r:.2f}")
    print(f"  预期: 基础3.0 - 0.0 + 效率(1/1)×2 ≈ 5.0")


# ==================== 主程序 ====================
if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='掌心战争 收益导向 DQN 训练')
    p.add_argument('--debug',    action='store_true',
                   help='运行奖励函数调试并退出')
    p.add_argument('--warmup',   type=int,   default=1000,
                   help='预热局数')
    p.add_argument('--total',    type=int,   default=15000,
                   help='混合+精炼合计局数')
    p.add_argument('--sp_ratio', type=float, default=0.65,
                   help='Self-Play 占比')
    p.add_argument('--refine',   type=float, default=0.75,
                   help='精炼期开始比例')
    p.add_argument('--lr',       type=float, default=5e-4,
                   help='学习率')
    p.add_argument('--gamma',    type=float, default=0.98,
                   help='折扣因子')
    p.add_argument('--hidden',   type=int,   default=256,
                   help='隐藏层大小')
    p.add_argument('--no_per',   action='store_true',
                   help='禁用PER优先级经验池')
    p.add_argument('--snap',     type=int,   default=300,
                   help='快照保存间隔')
    p.add_argument('--passive',  type=str,   default='empty_city',
                   help='被动技能')
    p.add_argument('--model',    type=str,   default='output/dqn_mixed_v2.pt',
                   help='模型保存路径')
    p.add_argument('--eval_g',   type=int,   default=500,
                   help='评估局数')
    args = p.parse_args()

    if args.debug:
        debug_reward()
        exit(0)

    remain       = 1.0 - args.sp_ratio
    random_ratio = remain * 0.50
    greedy_ratio = remain * 0.33
    fixed_ratio  = remain * 0.17

    print(f"[混合训练] 设备={DEVICE}")
    print(f"  Self-Play={args.sp_ratio:.0%}  Random={random_ratio:.0%}  "
          f"Greedy={greedy_ratio:.0%}  Fixed={fixed_ratio:.0%}")
    print(f"  LR={args.lr}  gamma={args.gamma}  hidden={args.hidden}")
    print(f"  PER={'开启' if not args.no_per else '关闭'}")

    ag = train_mixed(
        warmup_eps      = args.warmup,
        total_eps       = args.total,
        random_ratio    = random_ratio,
        greedy_ratio    = greedy_ratio,
        fixed_ratio     = fixed_ratio,
        refine_start    = args.refine,
        save            = args.model,
        passive         = args.passive,
        lr              = args.lr,
        gamma           = args.gamma,
        hidden          = args.hidden,
        use_per         = not args.no_per,
    )

    for opp in ['random', 'greedy', 'fixed']:
        evaluate(ag, games=args.eval_g, opponent=opp)

    t = build_table(ag)
    for pen in range(7):
        for aen in range(7):
            print_heatmap(t, pEn=pen, aEn=aen)

    out = {
        'timestamp':  datetime.now().isoformat(),
        'episodes':   args.warmup + args.total,
        'best_table': t.tolist(),
        'skills':     SKILL_NAMES,
    }
    with open('output/best_action_table_v2.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[保存] output/best_action_table_v2.json")
