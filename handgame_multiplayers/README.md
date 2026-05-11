# handgame AI

这是一个双人控制台手牌游戏的 AI 训练与决策实现。

## 主要文件

- `game.py`：对局流程与交互
- `heros/basic.py`：角色与回合结算逻辑
- `ai/core.py`：遗传算法 + minimax 搜索 AI
- `ai/train.py`：AI 训练入口

## 训练参数

`ai/train.py` 暴露了这些可调参数：

- `--depth`：minimax 搜索树深度
- `--branch-limit`：每层保留的最大分支数
- `--max-rounds`：单局自博弈最大回合数
- `--population-size`：种群规模
- `--generations`：训练代数
- `--elite-rate`：精英保留率
- `--elimination-rate`：淘汰率
- `--mutation-temperature`：变异温度
- `--mutation-sigma`：变异标准差基数
- `--crossover-rate`：交叉概率
- `--tournament-size`：锦标赛选择规模
- `--games-per-fitness`：适应度评估对局数
- `--seed`：随机种子
- `--self-role`：训练中己方角色名称
- `--enemy-role`：训练中对方角色名称
- `--output`：输出文件名；留空时自动按双方角色命名

## 训练示例

```bash
python -m ai.train --depth 2 --population-size 24 --generations 20 --mutation-temperature 0.35 --elimination-rate 0.5 --self-role 玩家1 --enemy-role 玩家2
```

运行后会生成类似 `best_genome_玩家1_vs_玩家2.json` 的文件，里面保存当前最佳评分参数。
