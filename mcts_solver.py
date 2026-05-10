"""
MCTS 求解器：掌心战争（空城被动模式）

算法：Decoupled UCT (DUCT)，专为同步行动博弈设计
  - AI 方：UCB 取 max（追求最大收益）
  - 对手方：UCB 取 min（假设对手最优反制）
  - 双方统计独立，不耦合

默认初始局面：双方 3 血 2 能
  - 3 血：减小状态空间（4×7×4×7 = 784 节点）
  - 2 能：专家先验（前两回合攒能胜率最高）

训练产物：所有可达状态的策略表（最优技能查找表）
"""

import numpy as np
import random
import math
import json
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

# ─── 游戏常量 ────────────────────────────────────────────────────────────────
NUM_SKILLS = 9
MAX_EN     = 6

SKILL_NAMES = ['单枪', '双枪', '三枪', '大招', '小防', '大防', '能量', '反弹', '清零']
SKILL_COST  = np.array([1, 2, 2, 3, 0, 1, 0, 1, 2], dtype=np.int32)
SKILL_ATK   = np.array([1, 1, 2, 3, -1, -1, -1, -1, -1], dtype=np.int32)

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
    [0,0,1,1,1,1,1,1,1],
], dtype=np.int32)


# ─── 工具函数 ────────────────────────────────────────────────────────────────
def get_available(energy: int) -> List[int]:
    avail = [sk for sk in range(NUM_SKILLS) if SKILL_COST[sk] <= energy]
    return avail if avail else [6]


def step_state(pHP: int, pEn: int, aHP: int, aEn: int,
               p_sk: int, a_sk: int,
               passive: str = 'empty_city',
               max_hp: int = 3) -> Tuple[int, int, int, int]:
    """
    执行一回合，返回新状态 (pHP, pEn, aHP, aEn)。
    p = AI，a = 对手。
    """
    # 1. 能量结算
    np_en = max(0, pEn - int(SKILL_COST[p_sk]))
    na_en = max(0, aEn - int(SKILL_COST[a_sk]))
    if p_sk == 6: np_en += 1
    if a_sk == 6: na_en += 1
    np_en = min(np_en, MAX_EN)
    na_en = min(na_en, MAX_EN)

    # 2. 清零
    if ZERO_TABLE[p_sk][a_sk]: np_en = na_en = 0
    if ZERO_TABLE[a_sk][p_sk]: np_en = na_en = 0

    # 3. 伤害（HP_MATRIX[攻方][守方] = 守方 HP 变化，负=受伤）
    na_hp = max(0, min(max_hp, aHP + int(HP_MATRIX[p_sk][a_sk])))
    np_hp = max(0, min(max_hp, pHP + int(HP_MATRIX[a_sk][p_sk])))

    # 4. 死亡判定（被动前）
    if np_hp <= 0 or na_hp <= 0:
        return np_hp, np_en, na_hp, na_en

    # 5. 空城被动
    if passive == 'empty_city':
        p_atk = max(int(SKILL_ATK[p_sk]), 0)
        a_atk = max(int(SKILL_ATK[a_sk]), 0)

        # AI 空城（pEn=0，对手攻击）→ 对手受伤
        if np_en == 0 and a_atk > 0:
            na_hp -= 1
            if int(HP_MATRIX[a_sk][p_sk]) >= 0:  # AI 没受正常伤 → 超级空城
                na_hp -= 1
            na_hp = max(0, na_hp)

        # 对手空城（aEn=0，AI 攻击）→ AI 受伤
        if na_en == 0 and p_atk > 0:
            np_hp -= 1
            if int(HP_MATRIX[p_sk][a_sk]) >= 0:  # 对手没受正常伤 → 超级空城
                np_hp -= 1
            np_hp = max(0, np_hp)

    return np_hp, np_en, na_hp, na_en


def terminal_value(pHP: int, aHP: int) -> float:
    """从 AI 视角返回终局分值：+1=AI赢，-1=对手赢，0=平"""
    if pHP <= 0 and aHP <= 0: return 0.0
    if aHP <= 0: return 1.0   # AI 赢
    if pHP <= 0: return -1.0  # 对手赢
    return 0.0                # 超时平局


# ─── MCTS 节点（DUCT：双方统计独立）────────────────────────────────────────
class MCTSNode:
    """
    使用置换表（transposition table），同一状态在不同路径共享统计。

    ai_n[act]  : AI 选择技能 act 的次数
    ai_q[act]  : AI 选择技能 act 后的累计价值（从 AI 视角，越高越好）
    pl_n[act]  : 对手选择技能 act 的次数
    pl_q[act]  : 对手选择技能 act 后的累计价值（从 AI 视角）
    N          : 该节点总访问次数
    """
    __slots__ = ('N', 'ai_n', 'ai_q', 'pl_n', 'pl_q', 'ai_avail', 'pl_avail')

    def __init__(self, pEn: int, aEn: int):
        self.N        = 0
        self.ai_avail = get_available(pEn)
        self.pl_avail = get_available(aEn)
        self.ai_n     = np.zeros(NUM_SKILLS, dtype=np.int32)
        self.ai_q     = np.zeros(NUM_SKILLS, dtype=np.float64)
        self.pl_n     = np.zeros(NUM_SKILLS, dtype=np.int32)
        self.pl_q     = np.zeros(NUM_SKILLS, dtype=np.float64)


# ─── MCTS 求解器 ─────────────────────────────────────────────────────────────
class MCTSSolver:
    """
    DUCT-MCTS 求解器。

    train(n_sim) 后，所有可达状态均有策略估计。
    best_action(state) 返回 AI 的最优技能。
    """

    def __init__(self,
                 start_hp:  int   = 3,
                 start_en:  int   = 2,
                 passive:   str   = 'empty_city',
                 c_puct:    float = 1.4,
                 max_depth: int   = 40):
        self.start_state = (start_hp, start_en, start_hp, start_en)
        self.passive     = passive
        self.c_puct      = c_puct
        self.max_depth   = max_depth
        self.max_hp      = start_hp  # HP 上限与起始一致

        # 置换表：state -> MCTSNode
        self.nodes: Dict[Tuple[int,int,int,int], MCTSNode] = {}

    # ── 置换表访问 ──────────────────────────────────────────────────────────
    def _get_node(self, state: Tuple[int,int,int,int]) -> MCTSNode:
        if state not in self.nodes:
            _, pEn, _, aEn = state
            self.nodes[state] = MCTSNode(pEn, aEn)
        return self.nodes[state]

    # ── DUCT 动作选择 ────────────────────────────────────────────────────────
    def _select_ai(self, node: MCTSNode) -> int:
        """AI 用 UCB 取 max（最大化自身价值）"""
        log_N = math.log(node.N + 1)
        best_score, best_act = -math.inf, node.ai_avail[0]
        for act in node.ai_avail:
            n = node.ai_n[act]
            if n == 0:
                return act  # 未尝试的动作优先探索
            q = node.ai_q[act] / n
            u = self.c_puct * math.sqrt(log_N / n)
            score = q + u
            if score > best_score:
                best_score, best_act = score, act
        return best_act

    def _select_pl(self, node: MCTSNode) -> int:
        """对手用 UCB 取 min（最小化 AI 价值 = 最大化 -Q）"""
        log_N = math.log(node.N + 1)
        best_score, best_act = -math.inf, node.pl_avail[0]
        for act in node.pl_avail:
            n = node.pl_n[act]
            if n == 0:
                return act
            q = node.pl_q[act] / n
            u = self.c_puct * math.sqrt(log_N / n)
            score = -q + u  # 对手反向优化
            if score > best_score:
                best_score, best_act = score, act
        return best_act

    # ── 随机 Rollout ─────────────────────────────────────────────────────────
    def _rollout(self, pHP: int, pEn: int, aHP: int, aEn: int) -> float:
        """从当前状态随机对弈到终局，返回价值。"""
        for _ in range(self.max_depth):
            if pHP <= 0 or aHP <= 0:
                break
            ai_sk = random.choice(get_available(pEn))
            pl_sk = random.choice(get_available(aEn))
            pHP, pEn, aHP, aEn = step_state(
                pHP, pEn, aHP, aEn, ai_sk, pl_sk, self.passive, self.max_hp)
        return terminal_value(pHP, aHP)

    # ── 单次模拟 ─────────────────────────────────────────────────────────────
    def _simulate(self) -> float:
        """一次完整的 MCTS 模拟：选择 → rollout → 反向传播"""
        path: List[Tuple[Tuple, int, int]] = []  # [(state, ai_act, pl_act)]
        state = self.start_state
        pHP, pEn, aHP, aEn = state

        for _ in range(self.max_depth):
            if pHP <= 0 or aHP <= 0:
                value = terminal_value(pHP, aHP)
                self._backprop(path, value)
                return value

            node = self._get_node(state)

            if node.N == 0:
                # 叶节点：rollout 估算价值
                value = self._rollout(pHP, pEn, aHP, aEn)
                node.N = 1  # 标记已访问，下次走 UCB
                self._backprop(path, value)
                return value

            # 使用 DUCT 选择动作
            ai_act = self._select_ai(node)
            pl_act = self._select_pl(node)
            path.append((state, ai_act, pl_act))

            pHP, pEn, aHP, aEn = step_state(
                pHP, pEn, aHP, aEn, ai_act, pl_act, self.passive, self.max_hp)
            state = (pHP, pEn, aHP, aEn)

        # 超时平局
        value = terminal_value(pHP, aHP)
        self._backprop(path, value)
        return value

    # ── 反向传播 ─────────────────────────────────────────────────────────────
    def _backprop(self, path: list, value: float):
        for state, ai_act, pl_act in path:
            node = self._get_node(state)
            node.N            += 1
            node.ai_n[ai_act] += 1
            node.ai_q[ai_act] += value
            node.pl_n[pl_act] += 1
            node.pl_q[pl_act] += value

    # ── 训练入口 ─────────────────────────────────────────────────────────────
    def train(self, n_sim: int = 100_000, report_every: int = 10_000):
        """
        运行 n_sim 次模拟，逐步建立策略表。

        参数:
            n_sim        : 总模拟次数（越多越准，10万次约需几秒）
            report_every : 每多少次打印一次进度
        """
        print(f"\n[MCTS 训练] 起始局面: {self.start_state}  被动: {self.passive}")
        print(f"  模拟次数={n_sim}  c_puct={self.c_puct}")

        wins = draws = losses = 0
        for i in range(1, n_sim + 1):
            v = self._simulate()
            if   v > 0: wins   += 1
            elif v < 0: losses += 1
            else:       draws  += 1

            if i % report_every == 0:
                total = wins + draws + losses
                wr = wins / total
                n_nodes = len(self.nodes)
                root_n  = self.nodes.get(self.start_state)
                root_visits = root_n.N if root_n else 0
                print(f"  [{i:>7d}] AI胜率={wr:.1%}  "
                      f"节点数={n_nodes}  根节点访问={root_visits}")

        print(f"  [完成] 覆盖状态数: {len(self.nodes)}")

    # ── 查询接口 ─────────────────────────────────────────────────────────────
    def best_action(self, pHP: int, pEn: int, aHP: int, aEn: int) -> int:
        """
        返回当前局面 AI 的最优技能（访问次数最多的动作）。
        若该状态未被训练覆盖，fallback 到访问次数最多的可用动作。
        """
        if pHP <= 0 or aHP <= 0:
            return -1

        state = (pHP, pEn, aHP, aEn)
        if state not in self.nodes or self.nodes[state].N == 0:
            # 未覆盖状态：贪心选择（优先攻击，能量不足选充能）
            return self._greedy_fallback(pHP, pEn, aHP, aEn)

        node = self.nodes[state]
        avail = get_available(pEn)
        # 在可用动作中选访问次数最多的
        best_act = max(avail, key=lambda a: node.ai_n[a])
        return best_act

    def _greedy_fallback(self, pHP: int, pEn: int, aHP: int, aEn: int) -> int:
        """未覆盖状态的回退策略：优先选能打出最大伤害的技能。"""
        avail = get_available(pEn)
        best_act, best_dmg = avail[0], -999
        for a in avail:
            # 对能力范围内所有对手动作取平均伤害
            pl_avail = get_available(aEn)
            avg_dmg = sum(
                -int(HP_MATRIX[a][pl_sk]) for pl_sk in pl_avail
            ) / len(pl_avail)
            if avg_dmg > best_dmg:
                best_dmg, best_act = avg_dmg, a
        return best_act

    def state_confidence(self, pHP: int, pEn: int, aHP: int, aEn: int) -> int:
        """返回该状态的访问次数（越高越可信）。"""
        state = (pHP, pEn, aHP, aEn)
        return self.nodes[state].N if state in self.nodes else 0

    def state_win_rate(self, pHP: int, pEn: int, aHP: int, aEn: int) -> float:
        """返回该状态 AI 的胜率估计（基于 MCTS 统计）。"""
        state = (pHP, pEn, aHP, aEn)
        if state not in self.nodes:
            return 0.5
        node = self.nodes[state]
        if node.N == 0:
            return 0.5
        total_q = sum(
            node.ai_q[a] for a in get_available(state[1]) if node.ai_n[a] > 0
        )
        total_n = sum(
            node.ai_n[a] for a in get_available(state[1])
        )
        return (total_q / total_n + 1) / 2 if total_n > 0 else 0.5

    # ── 策略表打印 ────────────────────────────────────────────────────────────
    def print_policy(self, pHP: Optional[int] = None,
                     aHP: Optional[int] = None,
                     show_visits: bool = False):
        """
        打印 AI 策略表。
        pHP/aHP = None 打印所有血量。
        show_visits = True 时显示访问次数而非技能名。
        """
        php_list = [pHP] if pHP is not None else list(range(1, self.max_hp + 1))
        ahp_list = [aHP] if aHP is not None else list(range(1, self.max_hp + 1))

        for ph in php_list:
            for ah in ahp_list:
                wr_sum = sum(
                    self.state_win_rate(ph, pe, ah, ae)
                    for pe in range(MAX_EN + 1)
                    for ae in range(MAX_EN + 1)
                ) / ((MAX_EN + 1) ** 2)
                print(f"\n{'─'*56}")
                print(f"  AI血={ph}  对手血={ah}  "
                      f"(平均胜率估计: {wr_sum:.1%})")
                print(f"  {'AI能\\对手能':>10s}", end="")
                for ae in range(MAX_EN + 1):
                    print(f"  [{ae}能]", end="")
                print()

                for pe in range(MAX_EN + 1):
                    print(f"  AI [{pe}能]  ", end="")
                    for ae in range(MAX_EN + 1):
                        state = (ph, pe, ah, ae)
                        if state not in self.nodes or self.nodes[state].N == 0:
                            print(f"  {'?':>4s} ", end="")
                        elif show_visits:
                            n = self.nodes[state].N
                            print(f"  {n:>4d} ", end="")
                        else:
                            act  = self.best_action(ph, pe, ah, ae)
                            name = SKILL_NAMES[act] if act >= 0 else '  -  '
                            conf = self.nodes[state].N
                            mark = ' ' if conf > 100 else '~'  # ~ = 低置信度
                            print(f"  {name:>4s}{mark}", end="")
                    print()

        print(f"\n  技能: " + "  ".join(
            f"{i}:{n}" for i, n in enumerate(SKILL_NAMES)))
        print(f"  (~=访问次数<100，置信度低)")

    def top_states(self, n: int = 15):
        """列出访问次数最多的 top-n 状态及其策略。"""
        ranked = sorted(
            self.nodes.items(),
            key=lambda kv: kv[1].N,
            reverse=True
        )[:n]
        print(f"\n[访问最频繁的 {n} 个状态]")
        for (ph, pe, ah, ae), node in ranked:
            act  = self.best_action(ph, pe, ah, ae)
            name = SKILL_NAMES[act] if act >= 0 else '-'
            wr   = self.state_win_rate(ph, pe, ah, ae)
            print(f"  AI({ph}血{pe}能) vs 对手({ah}血{ae}能):  "
                  f"访问={node.N:>6d}  胜率≈{wr:.1%}  最优技能={name}")

    # ── 对战评估 ─────────────────────────────────────────────────────────────
    def evaluate(self, n_games: int = 2000,
                 opponent: str = 'random') -> float:
        """
        用训练好的策略打 n_games 局，统计 AI 胜率。
        opponent: 'random' | 'greedy' | 'self'（MCTS自博弈）
        """
        wins = draws = losses = 0
        for _ in range(n_games):
            pHP, pEn, aHP, aEn = self.start_state
            for _ in range(self.max_depth):
                if pHP <= 0 or aHP <= 0:
                    break
                # AI 用训练策略
                ai_sk = self.best_action(pHP, pEn, aHP, aEn)

                # 对手策略
                pl_avail = get_available(aEn)
                if opponent == 'random':
                    pl_sk = random.choice(pl_avail)
                elif opponent == 'greedy':
                    ai_avail = get_available(pEn)
                    best_a, best_dmg = pl_avail[0], 999
                    for a in pl_avail:
                        dmg = int(HP_MATRIX[a][ai_sk])
                        if dmg < best_dmg:
                            best_dmg, best_a = dmg, a
                    pl_sk = best_a
                elif opponent == 'self':
                    # 反视角查询（AI 和对手互换）
                    pl_sk = self.best_action(aHP, aEn, pHP, pEn)
                    if pl_sk < 0:
                        pl_sk = random.choice(pl_avail)
                else:
                    pl_sk = random.choice(pl_avail)

                pHP, pEn, aHP, aEn = step_state(
                    pHP, pEn, aHP, aEn, ai_sk, pl_sk,
                    self.passive, self.max_hp)

            v = terminal_value(pHP, aHP)
            if   v > 0: wins   += 1
            elif v < 0: losses += 1
            else:       draws  += 1

        total = wins + draws + losses
        wr = wins / total
        print(f"  [评估 vs {opponent:>6s}] {n_games}局: "
              f"胜={wins}({wr:.1%})  平={draws}  负={losses}")
        return wr

    # ── 保存 / 加载 ──────────────────────────────────────────────────────────
    def save(self, path: str = 'output/mcts_policy.json'):
        """将训练好的策略（访问次数最多的动作）保存为查找表。"""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # 只存策略表（节省空间）
        policy = {}
        for (ph, pe, ah, ae), node in self.nodes.items():
            act = self.best_action(ph, pe, ah, ae)
            wr  = self.state_win_rate(ph, pe, ah, ae)
            policy[f"{ph},{pe},{ah},{ae}"] = {
                'action': int(act),
                'visits': int(node.N),
                'win_rate': round(wr, 4),
            }

        data = {
            'start_state':  list(self.start_state),
            'max_hp':       self.max_hp,
            'passive':      self.passive,
            'n_states':     len(self.nodes),
            'skill_names':  SKILL_NAMES,
            'policy':       policy,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [保存] 策略表 → {path}  ({len(policy)} 个状态)")

    def load(self, path: str = 'output/mcts_policy.json'):
        """加载已保存的策略表（只有策略，不含 MCTS 树）。"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.max_hp = data.get('max_hp', 3)
        self.passive = data.get('passive', 'empty_city')

        # 重建节点（仅填访问次数，供 best_action 使用）
        for key, val in data['policy'].items():
            ph, pe, ah, ae = map(int, key.split(','))
            state = (ph, pe, ah, ae)
            node  = self._get_node(state)
            act   = val['action']
            visits = val['visits']
            node.N         = visits
            node.ai_n[act] = visits  # 最优动作占全部访问
            node.ai_q[act] = visits * (val['win_rate'] * 2 - 1)

        print(f"  [加载] 策略表 ← {path}  ({len(data['policy'])} 个状态)")


# ─── 与 handgame.py / DQNAgent 兼容的包装类 ─────────────────────────────────
class MCTSAgent:
    """
    MCTSSolver 的包装，接口兼容现有 DQNAgent。
    可直接替换 handgame.py 中的 agent。

    用法:
        agent = MCTSAgent(solver)
        sk = agent.act(sid, available_actions=[...])
    """

    def __init__(self, solver: MCTSSolver):
        self.solver  = solver
        self.history = []

    @classmethod
    def load(cls, path: str = 'output/mcts_policy.json') -> 'MCTSAgent':
        solver = MCTSSolver()
        solver.load(path)
        return cls(solver)

    def act(self, sid: int, train: bool = False,
            available_actions: Optional[List[int]] = None) -> int:
        """
        sid: 与 dqn_train.py 一致的状态 ID：(pHP*7+pEn)*35 + (aHP*7+aEn)
        """
        aHP_aEn = sid % 35
        pHP_pEn = sid // 35
        aEn = aHP_aEn % 7;  aHP = aHP_aEn // 7
        pEn = pHP_pEn % 7;  pHP = pHP_pEn // 7

        act = self.solver.best_action(pHP, pEn, aHP, aEn)

        # 若最优动作不在可用列表，从可用动作中按 Q 值选最大
        if available_actions is not None and act not in available_actions:
            state = (pHP, pEn, aHP, aEn)
            if state in self.solver.nodes:
                node = self.solver.nodes[state]
                act  = max(available_actions,
                           key=lambda a: (node.ai_q[a] / max(node.ai_n[a], 1)))
            else:
                act = available_actions[0]

        return act

    # 空接口：兼容 DQNAgent
    def remember(self, *args): pass
    def replay(self):          pass
    def update_target(self):   pass
    def decay(self):           pass


# ─── 主程序 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='MCTS 掌心战争求解器')
    ap.add_argument('--hp',      type=int,   default=3,
                    help='双方起始血量（默认3）')
    ap.add_argument('--en',      type=int,   default=2,
                    help='双方起始能量（默认2）')
    ap.add_argument('--sim',     type=int,   default=200_000,
                    help='MCTS 模拟次数（默认20万）')
    ap.add_argument('--c',       type=float, default=1.4,
                    help='UCB 探索系数 c_puct（默认1.4）')
    ap.add_argument('--passive', default='empty_city',
                    choices=['empty_city', 'none'],
                    help='被动技能')
    ap.add_argument('--save',    default='output/mcts_policy.json',
                    help='策略保存路径')
    ap.add_argument('--eval_g',  type=int,   default=2000,
                    help='评估局数')
    ap.add_argument('--top',     type=int,   default=15,
                    help='显示访问最多的 top-N 状态')
    args = ap.parse_args()

    solver = MCTSSolver(
        start_hp  = args.hp,
        start_en  = args.en,
        passive   = args.passive,
        c_puct    = args.c,
    )

    # ── 训练 ─────────────────────────────────────────────────────────────
    solver.train(n_sim=args.sim, report_every=args.sim // 10)

    # ── 评估 ─────────────────────────────────────────────────────────────
    print("\n[评估]")
    solver.evaluate(n_games=args.eval_g, opponent='random')
    solver.evaluate(n_games=args.eval_g, opponent='greedy')
    solver.evaluate(n_games=args.eval_g, opponent='self')

    # ── 策略展示 ─────────────────────────────────────────────────────────
    print("\n[AI 对手 1 血时的策略]")
    solver.print_policy(aHP=1)

    print("\n[AI 对手 2 血时的策略]")
    solver.print_policy(aHP=2)

    print("\n[AI 对手 3 血时的策略]")
    solver.print_policy(aHP=3)

    solver.top_states(n=args.top)

    # ── 保存 ─────────────────────────────────────────────────────────────
    solver.save(args.save)

    print(f"\n[完成] 在 handgame.py 中替换 DQNAgent:")
    print(f"  from mcts_solver import MCTSAgent")
    print(f"  agent = MCTSAgent.load('{args.save}')")
