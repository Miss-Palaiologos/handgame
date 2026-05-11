from __future__ import annotations

import argparse

from .core import GeneticConfig, GeneticTrainer, SearchConfig, TrainingIdentity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the handgame AI with genetic self-play.")
    parser.add_argument("--depth", type=int, default=2, help="minimax 搜索树深度（按回合计）")
    parser.add_argument("--branch-limit", type=int, default=11, help="每层保留的最大分支数")
    parser.add_argument("--max-rounds", type=int, default=30, help="单局自博弈的最大回合数")
    parser.add_argument("--population-size", type=int, default=24, help="种群规模")
    parser.add_argument("--generations", type=int, default=20, help="训练代数")
    parser.add_argument("--elite-rate", type=float, default=0.2, help="精英保留率")
    parser.add_argument("--elimination-rate", type=float, default=0.5, help="淘汰率")
    parser.add_argument("--mutation-temperature", type=float, default=0.35, help="变异温度")
    parser.add_argument("--mutation-sigma", type=float, default=1.0, help="变异标准差基数")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="交叉概率")
    parser.add_argument("--tournament-size", type=int, default=3, help="锦标赛选择规模")
    parser.add_argument("--games-per-fitness", type=int, default=6, help="每个个体的适应度评估对局数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--self-role", type=str, default="玩家1", help="训练中己方角色名称")
    parser.add_argument("--enemy-role", type=str, default="玩家2", help="训练中对方角色名称")
    parser.add_argument("--output", type=str, default="", help="最佳基因输出文件；留空则自动按双方角色命名")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    search = SearchConfig(
        depth=args.depth,
        branch_limit=args.branch_limit,
        max_rounds=args.max_rounds,
    )
    config = GeneticConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_rate=args.elite_rate,
        elimination_rate=args.elimination_rate,
        mutation_temperature=args.mutation_temperature,
        mutation_sigma=args.mutation_sigma,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
        games_per_fitness=args.games_per_fitness,
        seed=args.seed,
    )
    identity = TrainingIdentity(self_role=args.self_role, enemy_role=args.enemy_role)

    trainer = GeneticTrainer(config, search, identity)
    best_genome, history = trainer.train()

    output_path = trainer.save_best(best_genome, args.output or None)

    print("训练完成")
    print(f"最佳基因已保存到: {output_path}")
    print("最佳基因参数:")
    print(best_genome.to_dict())
    print("最后一代训练摘要:")
    print(history[-1] if history else {})


if __name__ == "__main__":
    main()
