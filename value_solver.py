"""
HandGame 状态图价值求解器

状态空间: (pHP, pEn, aHP, aEn) = 5×7×5×7 = 1225 个节点
  p = 被训练的 AI
  a = 对手（玩家）

算法: Minimax 价值迭代
  V[s] = max_{AI技能} min_{对手技能} gamma * V[下一状态]
  - AI 方向取 max（追求最好结果）
  - 对手方向取 min（假设对手会做最优反制）

手动锚点: 对任意局面指定固定分数，迭代时不覆盖这些节点，
  算法从锚点向外传播。

使用示例:
    solver = HandGameSolver()
    # 我方2能，对方1血0能 → 必杀，打高分
    solver.set_score(pEn=2, aHP=1, aEn=0, value=90)
    solver.solve()
    solver.print_table(aHP=1)
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional, List

# ─── 游戏常量（与 dqn_train.py 保持一致）────────────────────────────────
NUM_SKILLS = 9
MAX_HP     = 4
MAX_EN     = 6

SKILL_NAMES = ['单枪', '双枪', '三枪', '大招', '小防', '大防', '能量', '反弹', '清零']
SKILL_COST  = np.array([1, 2, 2, 3, 0, 1, 0, 1, 2], dtype=np.int32)
SKILL_ATK   = np.array([1, 1, 2, 3, -1, -1, -1, -1, -1], dtype=np.int32)

# HP_MATRIX[attacker_skill][defender_skill] = 防守方 HP 变化（负=受伤）
HP_MATRIX = np.array([
    [  0,  -1,  -1,   0,   0,   0,  -1,   0,  -2],  # 单枪
    [  0,   0,   0,  -2,   0,   0,  -2,   0,  -3],  # 双枪
    [  0,  -1,   0,  -2,  -1,   0,  -3,  -1,   1],  # 三枪
    [ -3,   0,   0,   0,  -3,   0,  -3,  -3,   1],  # 大招
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],  # 小防
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],  # 大防
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],  # 能量
    [ -1,  -2,   0,   0,   0,   0,   0,   0,   1],  # 反弹
    [  0,   0,   0,   0,   0,   0,   0,   0,   1],  # 清零
], dtype=np.int32)

# ZERO_TABLE[user_skill][opponent_skill] = 清零是否成功
ZERO_TABLE = np.array([
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1],
], dtype=np.int32)

WIN_VALUE  =  100.0   # AI 赢
LOSE_VALUE = -100.0   # 对手赢
DRAW_VALUE =  -30.0   # 超时平局（对 AI 不利）


# ─── 状态转移核心 ───────────────────────────────────────────────────────
def step_state(pHP: int, pEn: int, aHP: int, aEn: int,
               p_skill: int, a_skill: int,
               passive: str = 'empty_city') -> Tuple[int, int, int, int]:
    """
    给定当前状态和双方技能，返回下一帧状态。
    p = AI（我方），a = 对手（玩家）
    """
    # 1. 能量结算
    np_en = max(0, pEn - int(SKILL_COST[p_skill]))
    na_en = max(0, aEn - int(SKILL_COST[a_skill]))
    if p_skill == 6: np_en += 1
    if a_skill == 6: na_en += 1
    np_en = min(np_en, MAX_EN)
    na_en = min(na_en, MAX_EN)

    # 2. 清零
    if ZERO_TABLE[p_skill][a_skill]: np_en = na_en = 0
    if ZERO_TABLE[a_skill][p_skill]: np_en = na_en = 0

    # 3. 伤害
    # HP_MATRIX[p_skill][a_skill] = 对手(a)的 HP 变化
    # HP_MATRIX[a_skill][p_skill] = AI(p)的 HP 变化
    na_hp = max(0, min(MAX_HP, aHP + int(HP_MATRIX[p_skill][a_skill])))
    np_hp = max(0, min(MAX_HP, pHP + int(HP_MATRIX[a_skill][p_skill])))

    # 4. 死亡判定（被动前）
    if np_hp <= 0 or na_hp <= 0:
        return np_hp, np_en, na_hp, na_en

    # 5. 空城被动
    if passive == 'empty_city':
        p_atk = max(int(SKILL_ATK[p_skill]), 0)
        a_atk = max(int(SKILL_ATK[a_skill]), 0)

        # AI 空城（pEn=0，对手攻击）→ 对手受伤
        if np_en == 0 and a_atk > 0:
            na_hp -= 1
            # 超级空城：AI 本回合没受正常伤（HP 变化 >= 0）
            if int(HP_MATRIX[a_skill][p_skill]) >= 0:
                na_hp -= 1
            na_hp = max(0, na_hp)

        # 对手空城（aEn=0，AI 攻击）→ AI 受伤
        if na_en == 0 and p_atk > 0:
            np_hp -= 1
            if int(HP_MATRIX[p_skill][a_skill]) >= 0:
                np_hp -= 1
            np_hp = max(0, np_hp)

    return np_hp, np_en, na_hp, na_en


def get_available(energy: int) -> List[int]:
    avail = [sk for sk in range(NUM_SKILLS) if SKILL_COST[sk] <= energy]
    return avail if avail else [6]  # 没能量时只能充能


# ─── 主求解器 ────────────────────────────────────────────────────────────
class HandGameSolver:
    """
    Minimax 价值迭代求解器。

    节点：所有 (pHP, pEn, aHP, aEn) 组合，共 1225 个。
    边：双方各出一个技能，状态转移到下一节点。
    值：正值对 AI 有利，负值对对手有利。

    用法：
        solver = HandGameSolver()
        solver.set_score(pEn=2, aHP=1, aEn=0, value=90)  # 添加锚点
        solver.solve()
        action = solver.best_action(pHP=3, pEn=2, aHP=1, aEn=0)
    """

    def __init__(self, passive: str = 'empty_city', gamma: float = 0.95):
        self.passive = passive
        self.gamma   = gamma

        shape = (MAX_HP + 1, MAX_EN + 1, MAX_HP + 1, MAX_EN + 1)
        self.V      = np.zeros(shape, dtype=np.float64)
        self.policy = np.full(shape, -1, dtype=np.int8)

        # 锚点集合：{(pHP, pEn, aHP, aEn): value}，求解时不会被覆盖
        self.anchors: Dict[Tuple[int, int, int, int], float] = {}

        # 强制技能：{(pHP, pEn, aHP, aEn): skill_index}
        # 价值迭代时只评估该技能，不做 max 选择
        self.forced_actions: Dict[Tuple[int, int, int, int], int] = {}

        self._init_terminals()

    # ── 终局初始化 ──────────────────────────────────────────────────────
    def _init_terminals(self):
        """将所有终局状态初始化为固定分值。"""
        for pHP in range(MAX_HP + 1):
            for aHP in range(MAX_HP + 1):
                if pHP <= 0 or aHP <= 0:
                    v = self._terminal_value(pHP, aHP)
                    for pEn in range(MAX_EN + 1):
                        for aEn in range(MAX_EN + 1):
                            self.V[pHP, pEn, aHP, aEn] = v
                            # 终局无技能可选
                            self.policy[pHP, pEn, aHP, aEn] = -1

    def _terminal_value(self, pHP: int, aHP: int) -> float:
        if pHP <= 0 and aHP <= 0: return DRAW_VALUE
        if aHP <= 0: return WIN_VALUE    # 对手血量 ≤ 0，AI 赢
        return LOSE_VALUE                # AI 血量 ≤ 0，对手赢

    def _is_terminal(self, pHP: int, aHP: int) -> bool:
        return pHP <= 0 or aHP <= 0

    # ── 锚点设置 ────────────────────────────────────────────────────────
    def set_score(self, value: float, *,
                  pHP: Optional[int] = None,
                  pEn: Optional[int] = None,
                  aHP: Optional[int] = None,
                  aEn: Optional[int] = None):
        """
        手动指定局面分数（锚点）。None 表示该维度匹配所有值。

        例：solver.set_score(90, pEn=2, aHP=1, aEn=0)
            → 所有「AI 2 能量，对手 1 血 0 能量」的局面设为 90 分
        """
        php_range = [pHP] if pHP is not None else range(1, MAX_HP + 1)
        pen_range = [pEn] if pEn is not None else range(MAX_EN + 1)
        ahp_range = [aHP] if aHP is not None else range(1, MAX_HP + 1)
        aen_range = [aEn] if aEn is not None else range(MAX_EN + 1)

        count = 0
        for ph in php_range:
            for pe in pen_range:
                for ah in ahp_range:
                    for ae in aen_range:
                        if self._is_terminal(ph, ah):
                            continue
                        key = (ph, pe, ah, ae)
                        self.anchors[key] = value
                        self.V[ph, pe, ah, ae] = value
                        count += 1

        desc_parts = []
        if pHP is not None: desc_parts.append(f"AI血={pHP}")
        if pEn is not None: desc_parts.append(f"AI能={pEn}")
        if aHP is not None: desc_parts.append(f"对手血={aHP}")
        if aEn is not None: desc_parts.append(f"对手能={aEn}")
        desc = "  ".join(desc_parts) or "（所有局面）"
        print(f"  [锚点] {desc} → {value:+.1f}  (共 {count} 个节点)")

    def set_forced_action(self, skill: int, *,
                          pHP: Optional[int] = None,
                          pEn: Optional[int] = None,
                          aHP: Optional[int] = None,
                          aEn: Optional[int] = None):
        """
        强制指定局面使用某技能（策略锁定）。
        能量不足时自动跳过，并提示需要几能才能使用。

        例：solver.set_forced_action(8, pHP=1, aEn=3)
            → AI 1血、对手3能，只要AI有≥2能就锁定用清零
            （pEn<2 的局面自动跳过，不受影响）
        """
        php_range = [pHP] if pHP is not None else range(1, MAX_HP + 1)
        pen_range = [pEn] if pEn is not None else range(MAX_EN + 1)
        ahp_range = [aHP] if aHP is not None else range(1, MAX_HP + 1)
        aen_range = [aEn] if aEn is not None else range(MAX_EN + 1)

        cost     = int(SKILL_COST[skill])
        sk_name  = SKILL_NAMES[skill]
        set_cnt  = skip_energy = skip_anchor = 0

        for ph in php_range:
            for pe in pen_range:
                for ah in ahp_range:
                    for ae in aen_range:
                        if self._is_terminal(ph, ah):
                            continue
                        if pe < cost:
                            skip_energy += 1
                            continue
                        key = (ph, pe, ah, ae)
                        if key in self.anchors:
                            skip_anchor += 1
                            continue  # 锚点状态值固定，强制技能无意义
                        self.forced_actions[key] = skill
                        set_cnt += 1

        desc_parts = []
        if pHP is not None: desc_parts.append(f"AI血={pHP}")
        if pEn is not None: desc_parts.append(f"AI能={pEn}")
        if aHP is not None: desc_parts.append(f"对手血={aHP}")
        if aEn is not None: desc_parts.append(f"对手能={aEn}")
        desc = "  ".join(desc_parts) or "（所有局面）"

        notes = []
        if skip_energy:
            notes.append(f"能量不足跳过={skip_energy}个（需≥{cost}能）")
        if skip_anchor:
            notes.append(f"与锚点冲突跳过={skip_anchor}个")
        note_str = "  " + "  ".join(notes) if notes else ""
        print(f"  [强制技能] {sk_name}  {desc}  → 设置={set_cnt}个节点{note_str}")

    # ── 价值迭代 ─────────────────────────────────────────────────────────
    def solve(self, max_iter: int = 500, tol: float = 1e-5,
              method: str = 'maximin') -> int:
        """
        运行价值迭代直到收敛。

        method:
          'maximin'  - 悲观策略：AI 取 max，对手取 min（对手会做最优反制）
          'maxexpect' - 期望策略：AI 取 max，对手出招均匀随机

        返回: 收敛所用轮数
        """
        print(f"\n[求解] 方法={method}  gamma={self.gamma}  "
              f"锚点={len(self.anchors)}个  强制技能={len(self.forced_actions)}个节点")

        for it in range(max_iter):
            delta = 0.0

            for pHP in range(1, MAX_HP + 1):
                for pEn in range(MAX_EN + 1):
                    for aHP in range(1, MAX_HP + 1):
                        for aEn in range(MAX_EN + 1):
                            key = (pHP, pEn, aHP, aEn)
                            if key in self.anchors:
                                continue  # 锚点固定，不更新

                            old_v = self.V[pHP, pEn, aHP, aEn]
                            new_v, best_a = self._eval_state(
                                pHP, pEn, aHP, aEn, method)

                            self.V[pHP, pEn, aHP, aEn]      = new_v
                            self.policy[pHP, pEn, aHP, aEn] = best_a
                            delta = max(delta, abs(new_v - old_v))

            if (it + 1) % 50 == 0 or delta < tol:
                print(f"  迭代 {it+1:4d}: 最大变化量 = {delta:.6f}")

            if delta < tol:
                print(f"  [收敛] 共 {it + 1} 轮")
                return it + 1

        print(f"  [警告] {max_iter} 轮未收敛，最后变化量 = {delta:.6f}")
        return max_iter

    def _eval_state(self, pHP, pEn, aHP, aEn,
                    method: str) -> Tuple[float, int]:
        """
        对单个非锚点、非终局状态求最优动作和对应价值。
        若该状态有强制技能，只评估该技能（不做 max 选择）。
        返回 (value, best_ai_skill)
        """
        pl_avail = get_available(aEn)

        # 强制技能：跳过 max，只算该动作对应的价值
        forced = self.forced_actions.get((pHP, pEn, aHP, aEn))
        if forced is not None:
            avail = get_available(pEn)
            ai_sk = forced if forced in avail else avail[0]
            if method == 'maximin':
                worst_v = np.inf
                for pl_sk in pl_avail:
                    ns = step_state(pHP, pEn, aHP, aEn, ai_sk, pl_sk, self.passive)
                    nv = float(self.V[ns])
                    v  = nv if self._is_terminal(ns[0], ns[2]) else self.gamma * nv
                    worst_v = min(worst_v, v)
                return worst_v, ai_sk
            else:
                total = 0.0
                for pl_sk in pl_avail:
                    ns = step_state(pHP, pEn, aHP, aEn, ai_sk, pl_sk, self.passive)
                    nv = float(self.V[ns])
                    total += nv if self._is_terminal(ns[0], ns[2]) else self.gamma * nv
                return total / len(pl_avail), ai_sk

        ai_avail = get_available(pEn)

        best_v = -np.inf
        best_a = ai_avail[0]

        for ai_sk in ai_avail:
            if method == 'maximin':
                # 对手做最坏反制
                worst_v = np.inf
                for pl_sk in pl_avail:
                    ns = step_state(pHP, pEn, aHP, aEn, ai_sk, pl_sk,
                                    self.passive)
                    nv = float(self.V[ns])
                    v  = nv if self._is_terminal(ns[0], ns[2]) else self.gamma * nv
                    worst_v = min(worst_v, v)
                ai_score = worst_v

            elif method == 'maxexpect':
                # 对手均匀随机
                exp_v = 0.0
                for pl_sk in pl_avail:
                    ns = step_state(pHP, pEn, aHP, aEn, ai_sk, pl_sk,
                                    self.passive)
                    nv = float(self.V[ns])
                    v  = nv if self._is_terminal(ns[0], ns[2]) else self.gamma * nv
                    exp_v += v
                ai_score = exp_v / len(pl_avail)

            else:
                raise ValueError(f"未知 method: {method}")

            if ai_score > best_v:
                best_v = ai_score
                best_a = ai_sk

        return best_v, best_a

    # ── 查询接口 ─────────────────────────────────────────────────────────
    def best_action(self, pHP: int, pEn: int, aHP: int, aEn: int) -> int:
        """返回当前局面 AI 的最优技能编号，-1 表示游戏已结束。"""
        if self._is_terminal(pHP, aHP):
            return -1
        act = int(self.policy[pHP, pEn, aHP, aEn])
        # 如果策略还未计算（-1），fallback 到可用动作的第一个
        if act == -1:
            act = get_available(pEn)[0]
        return act

    def state_value(self, pHP: int, pEn: int, aHP: int, aEn: int) -> float:
        return float(self.V[pHP, pEn, aHP, aEn])

    def print_state(self, pHP: int, pEn: int, aHP: int, aEn: int):
        """打印单个局面的分析。"""
        v   = self.state_value(pHP, pEn, aHP, aEn)
        act = self.best_action(pHP, pEn, aHP, aEn)
        sk_name = SKILL_NAMES[act] if act >= 0 else '游戏结束'
        key = (pHP, pEn, aHP, aEn)
        if key in self.anchors:
            anchor = ' [锚点]'
        elif key in self.forced_actions:
            anchor = f' [强制:{SKILL_NAMES[self.forced_actions[key]]}]'
        else:
            anchor = ''
        print(f"  AI({pHP}血{pEn}能) vs 对手({aHP}血{aEn}能):  "
              f"值={v:+7.2f}  最优技能={sk_name}{anchor}")

    # ── 打印策略表 ───────────────────────────────────────────────────────
    def print_table(self, pHP: Optional[int] = None,
                    aHP: Optional[int] = None,
                    show_value: bool = False):
        """
        打印策略表（最优技能）。
        pHP/aHP = None 表示打印所有血量组合。
        show_value = True 时显示数值而非技能名。
        """
        php_list = [pHP] if pHP is not None else list(range(1, MAX_HP + 1))
        ahp_list = [aHP] if aHP is not None else list(range(1, MAX_HP + 1))

        for ph in php_list:
            for ah in ahp_list:
                tag = f"AI血={ph}  对手血={ah}"
                print(f"\n{'─'*50}")
                print(f"  {tag}")
                print(f"  {'AI能\\对手能':>10s}", end="")
                for ae in range(MAX_EN + 1):
                    print(f"  [{ae}能]", end="")
                print()
                for pe in range(MAX_EN + 1):
                    print(f"  AI [{pe}能]  ", end="")
                    for ae in range(MAX_EN + 1):
                        if show_value:
                            v = self.state_value(ph, pe, ah, ae)
                            print(f"  {v:+5.0f}", end="")
                        else:
                            act = self.best_action(ph, pe, ah, ae)
                            name = SKILL_NAMES[act] if act >= 0 else '  -  '
                            anchor_mark = '*' if (ph, pe, ah, ae) in self.anchors else ' '
                            print(f"  {name:>4s}{anchor_mark}", end="")
                    print()

        if not show_value:
            print(f"\n  (* = 手动锚点)")
            print(f"  技能编号: " + "  ".join(
                f"{i}:{n}" for i, n in enumerate(SKILL_NAMES)))

    # ── 可视化最有价值的局面 ────────────────────────────────────────────
    def top_states(self, n: int = 20, for_ai: bool = True):
        """列出价值最高（或最低）的 n 个非终局状态。"""
        states = []
        for pHP in range(1, MAX_HP + 1):
            for pEn in range(MAX_EN + 1):
                for aHP in range(1, MAX_HP + 1):
                    for aEn in range(MAX_EN + 1):
                        v = self.state_value(pHP, pEn, aHP, aEn)
                        states.append(((pHP, pEn, aHP, aEn), v))

        states.sort(key=lambda x: x[1], reverse=for_ai)
        label = "AI 最有利" if for_ai else "AI 最不利"
        print(f"\n[{label} top-{n} 局面]")
        for (ph, pe, ah, ae), v in states[:n]:
            act     = self.best_action(ph, pe, ah, ae)
            sk_name = SKILL_NAMES[act] if act >= 0 else '-'
            anchor  = '*' if (ph, pe, ah, ae) in self.anchors else ' '
            print(f"  {anchor}AI({ph}血{pe}能) vs 对手({ah}血{ae}能):  "
                  f"值={v:+7.2f}  → {sk_name}")

    # ── 保存 / 加载 ──────────────────────────────────────────────────────
    def save(self, path: str = 'output/value_table.json'):
        """将价值表和策略表保存到 JSON 文件。"""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'V':           self.V.tolist(),
            'policy':      self.policy.tolist(),
            'anchors':     {str(k): v for k, v in self.anchors.items()},
            'gamma':       self.gamma,
            'passive':     self.passive,
            'skill_names': SKILL_NAMES,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [保存] 价值表 → {path}")

    def load(self, path: str = 'output/value_table.json'):
        """从 JSON 文件加载价值表和策略表。"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.V      = np.array(data['V'],      dtype=np.float64)
        self.policy = np.array(data['policy'], dtype=np.int8)
        self.anchors = {eval(k): v for k, v in data['anchors'].items()}
        self.gamma   = data.get('gamma', 0.95)
        self.passive = data.get('passive', 'empty_city')
        print(f"  [加载] 价值表 ← {path}")


# ─── 与 handgame.py 的 DQNAgent 兼容的包装类 ─────────────────────────────
class ValueTableAgent:
    """
    替换 DQNAgent 的轻量包装。
    接口与 DQNAgent 相同，可直接插入 handgame.py。

    用法:
        agent = ValueTableAgent.from_solver(solver)
        # 或
        agent = ValueTableAgent.load('output/value_table.json')
    """

    def __init__(self, solver: HandGameSolver):
        self.solver  = solver
        self.history = []   # 保持与 DQNAgent 接口一致

    @classmethod
    def from_solver(cls, solver: HandGameSolver) -> 'ValueTableAgent':
        return cls(solver)

    @classmethod
    def load(cls, path: str = 'output/value_table.json') -> 'ValueTableAgent':
        solver = HandGameSolver()
        solver.load(path)
        return cls(solver)

    def act(self, sid: int, train: bool = False,
            available_actions=None) -> int:
        """
        sid: 状态 ID（与 dqn_train.py 兼容：(pHP*7+pEn)*35 + (aHP*7+aEn)）
        返回最优技能编号。
        """
        # 解码状态 ID
        aHP_aEn = sid % 35
        pHP_pEn = sid // 35
        aEn = aHP_aEn % 7
        aHP = aHP_aEn // 7
        pEn = pHP_pEn % 7
        pHP = pHP_pEn // 7

        act = self.solver.best_action(pHP, pEn, aHP, aEn)

        # 如果最优动作不在可用列表，退化到可用动作中价值最高的
        if available_actions and act not in available_actions:
            best_v = -np.inf
            best_a = available_actions[0]
            for a in available_actions:
                # 估算各可用技能的期望价值（均匀随机对手）
                total = 0.0
                pl_avail = get_available(aEn)
                for pl_sk in pl_avail:
                    ns = step_state(pHP, pEn, aHP, aEn, a, pl_sk,
                                    self.solver.passive)
                    total += float(self.solver.V[ns])
                avg = total / len(pl_avail)
                if avg > best_v:
                    best_v = avg
                    best_a = a
            act = best_a

        return act

    # 以下方法为空实现，保证接口兼容
    def remember(self, *args): pass
    def replay(self):          pass
    def update_target(self):   pass
    def decay(self):           pass


# ─── 主程序：配置锚点 + 求解 + 展示 ────────────────────────────────────
def build_default_solver(method: str = 'maximin') -> HandGameSolver:
    """
    内置常识性锚点配置，然后求解。
    你可以在这里增删锚点来改变 AI 的倾向。
    """
    solver = HandGameSolver(passive='empty_city', gamma=0.95)

    print("\n[锚点配置] 手动指定关键局面的分数")
    print("─" * 50)

    # ── 对手濒死，AI 有足够能量 → 必杀机会，高分 ──────────
    # 对方1血0能：AI 2能可用三枪（必杀）
    solver.set_score(100, pEn=2, aHP=1, aEn=0)
    # 对方1血0能：AI 3能可用大招（必杀）
    solver.set_score(100, pEn=3, aHP=1, aEn=0)
    # 对方1血0能：AI 1能可用单枪（对手0能时命中）


    # ── AI 濒死，对手有大量能量 → 危险局面 ──────────────
    # pEn=0 时清零费2能用不了，先标记为危险分值（迭代时仍自由选技能）
    solver.set_score(-70, pHP=1, pEn=0, aEn=3)
    solver.set_score(-60, pHP=1, pEn=0, aEn=2)
    # pEn≥2 时有能量用清零：锁定清零对抗对手大能量
    # 清零 vs 大招/三枪：成功，AI回血；清零 vs 单枪/双枪：失败且受伤
    # → 适合对手倾向出大招时用，minimax 下会保守评估最坏情况
    solver.set_forced_action(8, pHP=1, aEn=3)  # 清零，pEn<2 自动跳过
    solver.set_forced_action(8, pHP=1, aEn=2)  # 清零，pEn<2 自动跳过

    # ── 双方濒死，AI 能量优势 → 略有优势 ────────────────
    solver.set_score(30, pHP=1, pEn=2, aHP=1, aEn=0)

    print("─" * 50)
    solver.solve(method=method)
    return solver


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='HandGame 状态图价值求解器')
    ap.add_argument('--method',  default='maximin',
                    choices=['maximin', 'maxexpect'],
                    help='求解策略：maximin=悲观  maxexpect=期望')
    ap.add_argument('--gamma',   type=float, default=0.95,
                    help='折扣因子（0-1，越小越重视快速获胜）')
    ap.add_argument('--save',    default='output/value_table.json',
                    help='价值表保存路径')
    ap.add_argument('--show_value', action='store_true',
                    help='打印分值而非技能名')
    ap.add_argument('--top',     type=int, default=15,
                    help='显示 top-N 局面')
    args = ap.parse_args()

    solver = HandGameSolver(gamma=args.gamma)

    # ── 在这里添加你的锚点 ──────────────────────────────
    print("\n[锚点配置]")
    # 示例：我方2能，对方1血0能 → 必杀（三枪），打 90 分
    solver.set_score(90, pEn=2, aHP=1, aEn=0)
    solver.set_score(90, pEn=3, aHP=1, aEn=0)
    solver.set_score(70, pEn=1, aHP=1, aEn=0)
    solver.set_score(50, pEn=3, aHP=2, aEn=0)
    solver.set_score(-70, pHP=1, pEn=0, aEn=3)

    # ── 求解 ────────────────────────────────────────────
    solver.solve(method=args.method)

    # ── 展示结果 ─────────────────────────────────────────
    print("\n[对手1血时的 AI 策略]")
    solver.print_table(aHP=1, show_value=args.show_value)

    print("\n[对手2血时的 AI 策略]")
    solver.print_table(aHP=2, show_value=args.show_value)

    solver.top_states(n=args.top, for_ai=True)
    solver.top_states(n=args.top, for_ai=False)

    # ── 保存 ────────────────────────────────────────────
    solver.save(args.save)

    print(f"\n[完成] 可在 handgame.py 中替换 DQNAgent:")
    print(f"  from value_solver import ValueTableAgent")
    print(f"  agent = ValueTableAgent.load('{args.save}')")
