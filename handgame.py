"""掌心战争 - 空城单挑模式（人机对战 + AI 在线学习）"""
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

# 运行目录：exe 时为 exe 所在目录，脚本时为脚本所在目录
if getattr(sys, 'frozen', False):
    _BASE = os.path.dirname(sys.executable)
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _BASE)

try:
    import torch
    from dqn_train import (
        DQNAgent, DEVICE,
        NUM_SKILLS, MAX_HP, SKILL_NAMES, SKILL_COST,
        HP_MATRIX, ZERO_TABLE, SKILL_ATK,
    )
    import numpy as np
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请确保 dqn_train.py 在同目录，并已安装 PyTorch")
    input("按回车退出...")
    sys.exit(1)

MAX_ENERGY   = 6
MAX_ROUNDS   = 30
MODEL_PATH   = os.path.join(_BASE, 'output', 'dqn_mixed.pt')
HISTORY_PATH = os.path.join(_BASE, 'output', 'history.json')


# ─── 工具函数 ──────────────────────────────────────────────
def get_available(energy, is_player=True):
    result = [sk for sk in range(NUM_SKILLS) if SKILL_COST[sk] <= energy]
    return result if result else [6]


def ai_state_id(pHP, pEn, aHP, aEn):
    """从 AI 视角的状态 ID（AI 是 player，人类是 opponent）"""
    return (aHP * 7 + aEn) * 35 + (pHP * 7 + pEn)


# ─── 奖励（从 AI 视角） ────────────────────────────────────
def calc_ai_step_reward(pHP0, aHP0, pEn0, aEn0, pHP, aHP, pEn, aEn,
                        rnd, p_zero, a_zero):
    reward  = -float(rnd)
    p_loss  = pHP0 - pHP   # 人类掉血（AI有利）
    a_loss  = aHP0 - aHP   # AI掉血（AI不利）
    reward += (p_loss * 2.0) - (a_loss * 2.0)
    if not p_zero and aEn0 > 0 and p_loss > 0:
        reward += 0.5
    if a_zero:
        reward += 1.5
    if aEn == 0 and not p_zero:
        reward += 0.3
    return reward


def calc_ai_final_reward(winner, pHP, aHP):
    if winner == 2:   return  10.0 + aHP * 3   # AI 赢
    elif winner == 1: return -10.0 - pHP * 3   # 人类赢
    else:             return -50.0              # 平局


# ─── 模型加载/保存 ─────────────────────────────────────────
def load_agent(path=MODEL_PATH):
    agent = DQNAgent(eps_start=0.05, eps_end=0.05, batch_size=32)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        agent.net.load_state_dict(ckpt['policy'])
        agent.tgt.load_state_dict(ckpt['policy'])
        print(f"  已加载模型: {path}")
    else:
        print(f"  未找到 {path}，使用随机初始化")
    return agent


def save_agent(agent, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'policy': agent.net.state_dict(),
                'hist':   agent.history}, path)


# ─── 历史存档 ──────────────────────────────────────────────
def load_history(path=HISTORY_PATH):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_history(steps, outcome, path=HISTORY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    history = load_history(path)
    history.append({
        'timestamp': datetime.now().isoformat(),
        'outcome':   outcome,   # 1=人赢 -1=AI赢 0=平
        'steps':     steps,
    })
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f)


def replay_from_history(agent, history, max_games=200):
    """将历史对局经验注入 replay buffer，让 AI 从过去对局中学习"""
    recent = history[-max_games:]
    count  = 0
    for game in recent:
        outcome = game['outcome']
        winner_map = {1: 1, -1: 2, 0: 3}
        winner = winner_map.get(outcome, 3)
        steps  = game['steps']
        for step in steps:
            sid  = ai_state_id(step['pHP'],      step['pEn'],
                               step['aHP'],      step['aEn'])
            sid2 = ai_state_id(step['next_pHP'], step['next_pEn'],
                               step['next_aHP'], step['next_aEn'])
            done = step['done']
            if done:
                r = calc_ai_final_reward(winner,
                                         step['next_pHP'], step['next_aHP'])
            else:
                r = calc_ai_step_reward(
                    step['pHP'],  step['aHP'],  step['pEn'],  step['aEn'],
                    step['next_pHP'], step['next_aHP'],
                    step['next_pEn'], step['next_aEn'],
                    step['rnd'], step['p_zero'], step['a_zero'])
            agent.remember(sid, step['aSk'], r, sid2, done)
            count += 1
    # 注入后集中训练几轮
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
        extra = 1 if int(HP_MATRIX[aSk][pSk]) >= 0 else 0
        aHP   = max(0, aHP - 1 - extra)
    if aHP > 0 and aEn == 0 and pAtk > 0:
        extra = 1 if int(HP_MATRIX[pSk][aSk]) >= 0 else 0
        pHP   = max(0, pHP - 1 - extra)
    return pHP, aHP


# ─── 单局游戏 ──────────────────────────────────────────────
def play_game(agent, pattern=None):
    pHP = aHP = MAX_HP
    pEn = aEn = 0
    winner = 3
    rnd    = 1
    steps  = []

    for rnd in range(1, MAX_ROUNDS + 1):
        render(rnd, pHP, pEn, aHP, aEn)

        # AI 决策
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

        # 1. 能量结算
        pEn = min(MAX_ENERGY, max(0, pEn - int(SKILL_COST[pSk])) + (1 if pSk == 6 else 0))
        aEn = min(MAX_ENERGY, max(0, aEn - int(SKILL_COST[aSk])) + (1 if aSk == 6 else 0))

        # 2. 清零
        p_zero = int(ZERO_TABLE[pSk][aSk])
        a_zero = int(ZERO_TABLE[aSk][pSk])
        if p_zero: pEn = aEn = 0
        if a_zero: pEn = aEn = 0

        # 3. 伤害
        aHP = max(0, min(MAX_HP, aHP + int(HP_MATRIX[pSk][aSk])))
        pHP = max(0, min(MAX_HP, pHP + int(HP_MATRIX[aSk][pSk])))

        # 4. 判断结束（伤害后）
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
            sid2   = ai_state_id(pHP, pEn, aHP, aEn)
            reward = calc_ai_final_reward(winner, pHP, aHP)
            agent.remember(sid, aSk, reward, sid2, True)
            agent.replay()
            break

        # 5. 空城被动
        pHP, aHP = apply_passive(pHP, pEn, aHP, aEn, pSk, aSk)
        done = pHP <= 0 or aHP <= 0
        if done:
            if pHP <= 0 and aHP <= 0: winner = 3
            elif aHP <= 0:            winner = 1
            else:                     winner = 2

        # 6. 收集经验 & 训练
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
        agent.remember(sid, aSk, reward, sid2, done)
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


# ─── 学习率调整 ────────────────────────────────────────────
BASE_LR = 1e-3
MAX_LR  = 5e-3
LR_STEP = 1e-3   # 每连败一局增加的 LR


def adjust_lr(agent, consec_losses):
    """连败越多 LR 越高，赢了归零恢复基础值"""
    lr = min(BASE_LR + consec_losses * LR_STEP, MAX_LR)
    for pg in agent.opt.param_groups:
        pg['lr'] = lr
    return lr


# ─── 主程序 ────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════╗")
    print("║       掌心战争  -  空城单挑模式          ║")
    print("║  技能: 0单枪 1双枪 2三枪 3大招           ║")
    print("║        4小防 5大防 6能量 7反弹 8清零     ║")
    print("║  规则: 双方均携带空城被动                ║")
    print("║  AI 将从每局对战中持续学习               ║")
    print("╚══════════════════════════════════════════╝\n")

    agent = load_agent(MODEL_PATH)

    # 加载历史存档，回放经验，分析人类出招模式
    history = load_history(HISTORY_PATH)
    if history:
        print(f"  历史存档: {len(history)} 局，正在回放经验...")
        n = replay_from_history(agent, history)
        print(f"  已注入 {n} 条历史经验到训练池")
    pattern = build_pattern(history)

    wins = draws = losses = 0
    consec_losses = 0
    game_n = 0

    while True:
        game_n += 1
        lr = adjust_lr(agent, consec_losses)
        print(f"\n  ── 第 {game_n} 局  (AI学习率 {lr:.0e}，历史{len(history)}局) ──")
        r, steps = play_game(agent, pattern)

        outcome = r  # 1=人赢 -1=AI赢 0=平
        if r == 1:
            wins += 1
            consec_losses += 1
        elif r == 0:
            draws += 1
            consec_losses = 0
        else:
            losses += 1
            consec_losses = 0

        save_history(steps, outcome, HISTORY_PATH)
        history.append({'outcome': outcome, 'steps': steps})
        pattern = build_pattern(history)

        total = wins + draws + losses
        loss_tip = f"  [{consec_losses}连败，LR已提升]" if consec_losses > 0 else ""
        print(f"  战绩: {total}局  {wins}胜 {draws}平 {losses}负  "
              f"胜率 {wins/total:.0%}{loss_tip}")

        save_agent(agent, MODEL_PATH)
        print(f"  模型已更新保存 → {MODEL_PATH}")

        again = input("\n  再来一局? (回车=是 / n=退出): ").strip().lower()
        if again == 'n':
            break

    print("\n  感谢游玩！")
    input("  按回车退出...")


if __name__ == '__main__':
    main()
