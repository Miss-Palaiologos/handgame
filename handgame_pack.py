"""掌心战争 - 空城单挑模式（打包专用，CPU推理，自包含）"""
import sys
import os
import json
from datetime import datetime

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

if getattr(sys, 'frozen', False):
    _BASE = os.path.dirname(sys.executable)
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import deque
    import random
    import copy
except ImportError as e:
    print(f"缺少依赖: {e}")
    input("按回车退出...")
    sys.exit(1)

DEVICE = torch.device('cpu')

# ─── 游戏常量 ─────────────────────────────────────────────
NUM_SKILLS = 9
MAX_HP     = 4
MAX_ENERGY = 6
MAX_ROUNDS = 30

SKILL_NAMES = ['单枪','双枪','三枪','大招','小防','大防','能量','反弹','清零']
SKILL_COST  = np.array([1, 2, 2, 3, 0, 1, 0, 1, 2], dtype=np.int32)

HP_MATRIX = np.array([
    [  0, -1, -1,  0,  0,  0, -1,  0, -2],
    [  0,  0,  0, -2,  0,  0, -2,  0, -3],
    [  0, -1,  0, -2, -1,  0, -3, -1,  1],
    [ -3,  0,  0,  0, -3,  0, -3, -3,  1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  1],
    [ -1, -2,  0,  0,  0,  0,  0,  0,  1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  1],
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

SKILL_ATK = np.array([1,1,2,3,-1,-1,-1,-1,-1], dtype=np.int32)

MODEL_PATH   = os.path.join(_BASE, 'output', 'dqn_mixed.pt')
HISTORY_PATH = os.path.join(_BASE, 'output', 'history.json')


# ─── 神经网络 ──────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim=1225, action_dim=9, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, action_dim),
        )
    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s_list, a_list, r_list, s2_list, d_list = zip(*batch)
        S  = np.array([_oh(s)  for s  in s_list],  dtype=np.float32)
        S2 = np.array([_oh(s2) for s2 in s2_list], dtype=np.float32)
        return S, list(a_list), list(r_list), S2, list(d_list)

    def __len__(self):
        return len(self.buf)


def _oh(sid, dim=1225):
    v = np.zeros(dim, dtype=np.float32)
    v[sid] = 1.0
    return v


class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.95,
                 eps_start=0.05, eps_end=0.05,
                 batch_size=32, memory=100_000, hidden=256):
        self.gamma    = gamma
        self.eps      = eps_start
        self.eps_end  = eps_end
        self.batch_sz = batch_size
        self.n_actions = NUM_SKILLS
        self.history  = []

        self.net = QNetwork(1225, 9, hidden).to(DEVICE)
        self.tgt = QNetwork(1225, 9, hidden).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.mem = ReplayMemory(memory)

    def act(self, sid, train=True, available_actions=None):
        if train and random.random() < self.eps:
            return random.choice(available_actions) if available_actions else random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            v = torch.FloatTensor([_oh(sid)]).to(DEVICE)
            q = self.net(v).squeeze(0)
            if available_actions is not None:
                mask = torch.full((self.n_actions,), float('-inf'), device=DEVICE)
                for a in available_actions:
                    mask[a] = 0
                q = q + mask
            return int(q.argmax(0).item())

    def remember(self, s, a, r, s2, d):
        self.mem.push(s, a, r, s2, d)

    def replay(self):
        if len(self.mem) < self.batch_sz:
            return
        S, A, R, S2, D = self.mem.sample(self.batch_sz)
        S_t  = torch.FloatTensor(S).to(DEVICE)
        A_t  = torch.LongTensor(A).to(DEVICE)
        R_t  = torch.FloatTensor(R).to(DEVICE)
        S2_t = torch.FloatTensor(S2).to(DEVICE)
        D_t  = torch.FloatTensor(D).to(DEVICE)
        q    = self.net(S_t).gather(1, A_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            tgt = R_t + (1 - D_t) * self.gamma * self.tgt(S2_t).max(1)[0]
        loss = F.mse_loss(q, tgt)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()


# ─── 工具函数 ──────────────────────────────────────────────
def get_available(energy, is_player=True):
    result = [sk for sk in range(NUM_SKILLS) if SKILL_COST[sk] <= energy]
    return result if result else [6]


def ai_state_id(pHP, pEn, aHP, aEn):
    return (aHP * 7 + aEn) * 35 + (pHP * 7 + pEn)


# ─── 奖励 ──────────────────────────────────────────────────
def calc_ai_step_reward(pHP0, aHP0, pEn0, aEn0, pHP, aHP, pEn, aEn,
                        rnd, p_zero, a_zero):
    reward  = -float(rnd)
    reward += (pHP0 - pHP) * 2.0 - (aHP0 - aHP) * 2.0
    if not p_zero and aEn0 > 0 and (pHP0 - pHP) > 0:
        reward += 0.5
    if a_zero:
        reward += 1.5
    if aEn == 0 and not p_zero:
        reward += 0.3
    return reward


def calc_ai_final_reward(winner, pHP, aHP):
    if winner == 2:   return  10.0 + aHP * 3
    elif winner == 1: return -10.0 - pHP * 3
    else:             return -50.0


# ─── 模型加载/保存 ─────────────────────────────────────────
def load_agent(path=MODEL_PATH):
    agent = DQNAgent()
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        agent.net.load_state_dict(ckpt['policy'])
        agent.tgt.load_state_dict(ckpt['policy'])
        print(f"  已加载模型: {path}")
    else:
        print(f"  未找到 {path}，使用随机初始化")
    return agent


def save_agent(agent, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'policy': agent.net.state_dict(), 'hist': agent.history}, path)


# ─── 历史存档 ──────────────────────────────────────────────
def load_history(path=HISTORY_PATH):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_history(steps, outcome, path=HISTORY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    history = load_history(path)
    history.append({'timestamp': datetime.now().isoformat(),
                    'outcome': outcome, 'steps': steps})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f)


def replay_from_history(agent, history, max_games=200):
    recent = history[-max_games:]
    count  = 0
    winner_map = {1: 1, -1: 2, 0: 3}
    for game in recent:
        winner = winner_map.get(game['outcome'], 3)
        for step in game['steps']:
            sid  = ai_state_id(step['pHP'],      step['pEn'],
                               step['aHP'],      step['aEn'])
            sid2 = ai_state_id(step['next_pHP'], step['next_pEn'],
                               step['next_aHP'], step['next_aEn'])
            done = step['done']
            r = (calc_ai_final_reward(winner, step['next_pHP'], step['next_aHP'])
                 if done else
                 calc_ai_step_reward(step['pHP'],  step['aHP'],
                                     step['pEn'],  step['aEn'],
                                     step['next_pHP'], step['next_aHP'],
                                     step['next_pEn'], step['next_aEn'],
                                     step['rnd'], step['p_zero'], step['a_zero']))
            agent.remember(sid, step['aSk'], r, sid2, done)
            count += 1
    for _ in range(min(count // 10 + 1, 50)):
        agent.replay()
    return count


DECAY = 0.9


def build_pattern(history):
    """Returns (p_pat, a_pat): {energy: {skill: weighted_count}}, 近期局权重更高."""
    p_pat, a_pat = {}, {}
    n = len(history)
    for i, game in enumerate(history):
        w = DECAY ** (n - 1 - i)
        for step in game['steps']:
            pEn, pSk = step['pEn'], step['pSk']
            aEn, aSk = step['aEn'], step['aSk']
            if pEn not in p_pat: p_pat[pEn] = {}
            p_pat[pEn][pSk] = p_pat[pEn].get(pSk, 0.0) + w
            if aEn not in a_pat: a_pat[aEn] = {}
            a_pat[aEn][aSk] = a_pat[aEn].get(aSk, 0.0) + w
    return p_pat, a_pat


def format_probs(pat, en, label):
    """列出某能量下所有技能的历史概率，按概率降序。"""
    if en not in pat or not pat[en]:
        return f"  [历史] {label}(能量{en}): 无记录"
    total = sum(pat[en].values())
    ranked = sorted(pat[en].items(), key=lambda x: -x[1])
    parts = [f"{SKILL_NAMES[sk]} {v/total:.0%}" for sk, v in ranked]
    return f"  [历史] {label}(能量{en}): {'  '.join(parts)}"


# ─── 界面 ──────────────────────────────────────────────────
def render(rnd, pHP, pEn, aHP, aEn):
    bar = lambda hp: '█' * hp + '░' * (MAX_HP - hp)
    print(f"\n{'─'*44}")
    print(f"  第 {rnd:2d} 回合")
    print(f"  你  [{bar(pHP)}]  HP={pHP}  能量={pEn}")
    print(f"  AI  [{bar(aHP)}]  HP={aHP}  能量={aEn}")
    print(f"{'─'*44}")


def pick_skill(avail):
    skills = '  '.join(f"{i}){SKILL_NAMES[i]}" for i in avail)
    print(f"  可用: {skills}")
    while True:
        try:
            sk = int(input("  选择编号: ").strip())
            if sk in avail:
                return sk
            print(f"  请从 {avail} 中选择")
        except (ValueError, EOFError):
            print("  请输入数字")


# ─── 空城被动 ──────────────────────────────────────────────
def apply_passive(pHP, pEn, aHP, aEn, pSk, aSk):
    pAtk = max(int(SKILL_ATK[pSk]), 0)
    aAtk = max(int(SKILL_ATK[aSk]), 0)
    if pHP > 0 and pEn == 0 and aAtk > 0:
        aHP = max(0, aHP - 1 - (1 if int(HP_MATRIX[aSk][pSk]) >= 0 else 0))
    if aHP > 0 and aEn == 0 and pAtk > 0:
        pHP = max(0, pHP - 1 - (1 if int(HP_MATRIX[pSk][aSk]) >= 0 else 0))
    return pHP, aHP


# ─── 学习率调整 ────────────────────────────────────────────
BASE_LR = 1e-3
MAX_LR  = 5e-3
LR_STEP = 1e-3


def adjust_lr(agent, consec_losses):
    lr = min(BASE_LR + consec_losses * LR_STEP, MAX_LR)
    for pg in agent.opt.param_groups:
        pg['lr'] = lr
    return lr


# ─── 单局游戏 ──────────────────────────────────────────────
def play_game(agent, pattern=None):
    pHP = aHP = MAX_HP
    pEn = aEn = 0
    winner = 3
    rnd    = 1
    steps  = []

    for rnd in range(1, MAX_ROUNDS + 1):
        render(rnd, pHP, pEn, aHP, aEn)

        sid     = ai_state_id(pHP, pEn, aHP, aEn)
        avail_a = get_available(aEn, is_player=False)
        aSk     = agent.act(sid, train=True, available_actions=avail_a)

        avail_p = get_available(pEn, is_player=True)
        pSk     = pick_skill(avail_p)

        print(f"\n  你: {SKILL_NAMES[pSk]}  vs  AI: {SKILL_NAMES[aSk]}")
        if pattern and pattern[0]:
            print(format_probs(pattern[0], pEn, '你'))
            print(format_probs(pattern[1], aEn, 'AI'))

        pHP0, aHP0, pEn0, aEn0 = pHP, aHP, pEn, aEn

        pEn = min(MAX_ENERGY, max(0, pEn - int(SKILL_COST[pSk])) + (1 if pSk == 6 else 0))
        aEn = min(MAX_ENERGY, max(0, aEn - int(SKILL_COST[aSk])) + (1 if aSk == 6 else 0))

        p_zero = int(ZERO_TABLE[pSk][aSk])
        a_zero = int(ZERO_TABLE[aSk][pSk])
        if p_zero: pEn = aEn = 0
        if a_zero: pEn = aEn = 0

        aHP = max(0, min(MAX_HP, aHP + int(HP_MATRIX[pSk][aSk])))
        pHP = max(0, min(MAX_HP, pHP + int(HP_MATRIX[aSk][pSk])))

        done = pHP <= 0 or aHP <= 0
        if done:
            if pHP <= 0 and aHP <= 0: winner = 3
            elif aHP <= 0:            winner = 1
            else:                     winner = 2
            steps.append({'pHP': pHP0, 'pEn': pEn0, 'aHP': aHP0, 'aEn': aEn0,
                          'pSk': pSk, 'aSk': aSk, 'rnd': rnd,
                          'next_pHP': pHP, 'next_pEn': pEn,
                          'next_aHP': aHP, 'next_aEn': aEn,
                          'p_zero': p_zero, 'a_zero': a_zero, 'done': True})
            agent.remember(ai_state_id(pHP0, pEn0, aHP0, aEn0), aSk,
                           calc_ai_final_reward(winner, pHP, aHP),
                           ai_state_id(pHP, pEn, aHP, aEn), True)
            agent.replay()
            break

        pHP, aHP = apply_passive(pHP, pEn, aHP, aEn, pSk, aSk)
        done = pHP <= 0 or aHP <= 0
        if done:
            if pHP <= 0 and aHP <= 0: winner = 3
            elif aHP <= 0:            winner = 1
            else:                     winner = 2

        steps.append({'pHP': pHP0, 'pEn': pEn0, 'aHP': aHP0, 'aEn': aEn0,
                      'pSk': pSk, 'aSk': aSk, 'rnd': rnd,
                      'next_pHP': pHP, 'next_pEn': pEn,
                      'next_aHP': aHP, 'next_aEn': aEn,
                      'p_zero': p_zero, 'a_zero': a_zero, 'done': done})
        sid2   = ai_state_id(pHP, pEn, aHP, aEn)
        reward = (calc_ai_final_reward(winner, pHP, aHP) if done
                  else calc_ai_step_reward(pHP0, aHP0, pEn0, aEn0,
                                           pHP, aHP, pEn, aEn,
                                           rnd, p_zero, a_zero))
        agent.remember(ai_state_id(pHP0, pEn0, aHP0, aEn0), aSk, reward, sid2, done)
        agent.replay()

        if done:
            break
    else:
        winner = 3

    render(rnd, pHP, pEn, aHP, aEn)
    if winner == 1:   msg, result = "你赢了！", 1
    elif winner == 2: msg, result = "AI 获胜！", -1
    else:             msg, result = "平局", 0
    print(f"\n  ══ {msg} ══\n")
    return result, steps


# ─── 主程序 ────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════╗")
    print("║       掌心战争  -  空城单挑模式          ║")
    print("║  技能: 0单枪 1双枪 2三枪 3大招           ║")
    print("║        4小防 5大防 6能量 7反弹 8清零     ║")
    print("║  规则: 双方均携带空城被动                ║")
    print("╚══════════════════════════════════════════╝\n")

    agent = load_agent(MODEL_PATH)

    history = load_history(HISTORY_PATH)
    if history:
        print(f"  历史存档: {len(history)} 局，正在回放经验...")
        n = replay_from_history(agent, history)
        print(f"  已注入 {n} 条历史经验")
    pattern = build_pattern(history)

    wins = draws = losses = 0
    consec_losses = 0
    game_n = 0

    while True:
        game_n += 1
        lr = adjust_lr(agent, consec_losses)
        print(f"\n  ── 第 {game_n} 局  (学习率 {lr:.0e}，历史 {len(history)} 局) ──")
        r, steps = play_game(agent, pattern)

        if r == 1:
            wins += 1;  consec_losses += 1
        elif r == 0:
            draws += 1; consec_losses = 0
        else:
            losses += 1; consec_losses = 0

        save_history(steps, r, HISTORY_PATH)
        history.append({'outcome': r, 'steps': steps})
        pattern = build_pattern(history)  # (p_pat, a_pat) with recency weighting

        total    = wins + draws + losses
        loss_tip = f"  [{consec_losses}连败，LR已提升]" if consec_losses > 0 else ""
        print(f"  战绩: {total}局  {wins}胜 {draws}平 {losses}负  "
              f"胜率 {wins/total:.0%}{loss_tip}")

        save_agent(agent, MODEL_PATH)
        print(f"  模型已保存 → {MODEL_PATH}")

        if input("\n  再来一局? (回车=是 / n=退出): ").strip().lower() == 'n':
            break

    print("\n  感谢游玩！")
    input("  按回车退出...")


if __name__ == '__main__':
    main()
