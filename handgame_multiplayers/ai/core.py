from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import json
import math
import random
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

from heros.basic import BasicHero, attack_action


ACTION_ORDER = {
    "self-attack": 0,
    "group-attack": 1,
    "anti-group-attack": 2,
    "attack": 3,
    "defense": 4,
    "control": 5,
    None: 6,
}


@dataclass(frozen=True)
class BattleSnapshot:
    hp: int
    hp_ceiling: int
    mp: int
    mp_ceiling: int
    damage_resistance: int
    shield: int
    revival_armor: int

    @classmethod
    def from_hero(cls, hero: BasicHero) -> "BattleSnapshot":
        return cls(
            hp=hero.hp,
            hp_ceiling=hero.hp_ceiling,
            mp=hero.mp,
            mp_ceiling=hero.mp_ceiling,
            damage_resistance=hero.damage_resistance,
            shield=hero.shield,
            revival_armor=hero.revival_armor,
        )


@dataclass
class BattleGenome:
    hp_weight: float = 12.0
    hp_ceiling_weight: float = 2.0
    mp_weight: float = 1.5
    mp_ceiling_weight: float = 0.3
    damage_resistance_weight: float = 1.2
    shield_weight: float = 3.0
    revival_armor_weight: float = 16.0
    bias: float = 0.0

    def score(self, me: BattleSnapshot, enemy: BattleSnapshot) -> float:
        return (
            self.bias
            + self.hp_weight * (me.hp - enemy.hp)
            + self.hp_ceiling_weight * (me.hp_ceiling - enemy.hp_ceiling)
            + self.mp_weight * (me.mp - enemy.mp)
            + self.mp_ceiling_weight * (me.mp_ceiling - enemy.mp_ceiling)
            + self.damage_resistance_weight * (me.damage_resistance - enemy.damage_resistance)
            + self.shield_weight * (me.shield - enemy.shield)
            + self.revival_armor_weight * (me.revival_armor - enemy.revival_armor)
        )

    @classmethod
    def random(cls, rng: random.Random, scale: float = 1.0) -> "BattleGenome":
        return cls(
            hp_weight=rng.uniform(4.0, 18.0) * scale,
            hp_ceiling_weight=rng.uniform(0.0, 6.0) * scale,
            mp_weight=rng.uniform(0.2, 4.0) * scale,
            mp_ceiling_weight=rng.uniform(0.0, 1.5) * scale,
            damage_resistance_weight=rng.uniform(0.0, 5.0) * scale,
            shield_weight=rng.uniform(0.0, 8.0) * scale,
            revival_armor_weight=rng.uniform(4.0, 24.0) * scale,
            bias=rng.uniform(-5.0, 5.0) * scale,
        )

    def mutated(self, rng: random.Random, temperature: float, sigma: float = 1.0) -> "BattleGenome":
        scale = max(0.0001, temperature) * sigma

        def mutate(value: float) -> float:
            return value + rng.gauss(0.0, scale)

        return BattleGenome(
            hp_weight=mutate(self.hp_weight),
            hp_ceiling_weight=mutate(self.hp_ceiling_weight),
            mp_weight=mutate(self.mp_weight),
            mp_ceiling_weight=mutate(self.mp_ceiling_weight),
            damage_resistance_weight=mutate(self.damage_resistance_weight),
            shield_weight=mutate(self.shield_weight),
            revival_armor_weight=mutate(self.revival_armor_weight),
            bias=mutate(self.bias),
        )

    @staticmethod
    def crossover(a: "BattleGenome", b: "BattleGenome", rng: random.Random) -> "BattleGenome":
        def choose(x: float, y: float) -> float:
            return x if rng.random() < 0.5 else y

        return BattleGenome(
            hp_weight=choose(a.hp_weight, b.hp_weight),
            hp_ceiling_weight=choose(a.hp_ceiling_weight, b.hp_ceiling_weight),
            mp_weight=choose(a.mp_weight, b.mp_weight),
            mp_ceiling_weight=choose(a.mp_ceiling_weight, b.mp_ceiling_weight),
            damage_resistance_weight=choose(a.damage_resistance_weight, b.damage_resistance_weight),
            shield_weight=choose(a.shield_weight, b.shield_weight),
            revival_armor_weight=choose(a.revival_armor_weight, b.revival_armor_weight),
            bias=choose(a.bias, b.bias),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "BattleGenome":
        # 兼容两种格式：
        # 1) 旧格式：文件根节点就是 genome 字段；
        # 2) 新格式：{"best_genome": {...}, "training_settings": {...}}
        if "best_genome" in payload and isinstance(payload["best_genome"], dict):
            payload = payload["best_genome"]
        elif "genome" in payload and isinstance(payload["genome"], dict):
            payload = payload["genome"]

        field_names = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in payload.items() if key in field_names}
        return cls(**kwargs)


@dataclass
class SearchConfig:
    depth: int = 2
    branch_limit: int = 11
    max_rounds: int = 30
    self_attack_clear_delay: bool = True


@dataclass
class GeneticConfig:
    population_size: int = 24
    generations: int = 20
    elite_rate: float = 0.2
    elimination_rate: float = 0.5
    mutation_temperature: float = 0.35
    mutation_sigma: float = 1.0
    crossover_rate: float = 0.7
    tournament_size: int = 3
    games_per_fitness: int = 6
    seed: Optional[int] = 42


GA_DEBUG_PRESETS: dict[str, GeneticConfig] = {
    "default": GeneticConfig(),
    "quick": GeneticConfig(
        population_size=8,
        generations=5,
        elite_rate=0.25,
        elimination_rate=0.5,
        mutation_temperature=0.45,
        games_per_fitness=3,
    ),
    "balanced": GeneticConfig(
        population_size=24,
        generations=20,
        elite_rate=0.2,
        elimination_rate=0.5,
        mutation_temperature=0.35,
        games_per_fitness=6,
    ),
    "deep": GeneticConfig(
        population_size=48,
        generations=40,
        elite_rate=0.15,
        elimination_rate=0.45,
        mutation_temperature=0.25,
        games_per_fitness=10,
    ),
}


def build_genetic_debug_config(profile: str = "default", **overrides) -> GeneticConfig:
    """构造遗传算法配置，便于快速调参。

    profile:
        可选: default / quick / balanced / deep
    overrides:
        任意 GeneticConfig 字段覆盖，例如 population_size=16。
    """

    if profile not in GA_DEBUG_PRESETS:
        raise KeyError(f"未知 profile: {profile}")

    base = asdict(GA_DEBUG_PRESETS[profile])
    valid_fields = {item.name for item in fields(GeneticConfig)}

    for key, value in overrides.items():
        if key not in valid_fields:
            raise KeyError(f"未知 GeneticConfig 字段: {key}")
        base[key] = value

    return GeneticConfig(**base)


@dataclass(frozen=True)
class TrainingIdentity:
    self_role: str = "玩家1"
    enemy_role: str = "玩家2"

    def slug(self) -> str:
        text = f"{self.self_role}_vs_{self.enemy_role}"
        text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text, flags=re.UNICODE)
        return text.strip("_") or "ai"


@dataclass(frozen=True)
class ActionPlan:
    action_id: int
    target_index: Optional[int]
    action_type: Optional[str]
    action_name: str


@dataclass(frozen=True)
class PlayerAIConfig:
    """单个玩家在对局中使用的 AI 配置。

    enabled:
        是否启用 AI 托管。
    genome:
        该 AI 的评分参数集合，决定局面评估倾向。
    search:
        该 AI 的搜索参数集合，决定 minimax 搜索深度与剪枝范围。
    """

    enabled: bool = True
    genome: BattleGenome = field(default_factory=BattleGenome)
    search: SearchConfig = field(default_factory=SearchConfig)
    label: str = "default"


def _clear_marked_actions(heroes: Sequence[BasicHero]) -> None:
    # 控制阶段结束后统一清理被标记的自戕对象。
    # 这样可以保证同一轮里多个反弹者都能看到原始自戕值。
    for hero in heroes:
        for stack in (hero.attack_stack, hero.defense_stack):
            for action in stack:
                if getattr(action, "marked_for_clear", False):
                    action.attack_value = 0
                    action.marked_for_clear = False


def _winner_name(heroes: Sequence[BasicHero]) -> Optional[str]:
    # 判断当前局面是否已经结束；这是 minimax 的终止条件之一。
    alive = [hero for hero in heroes if not hero.is_defeated()]
    if len(alive) == 2:
        return None
    if len(alive) == 1:
        return alive[0].name
    return "平局"


def _legal_actions(hero: BasicHero, enemy_index: int) -> list[ActionPlan]:
    # 根据当前 MP 生成本方所有合法动作。
    # 这里先抽取动作定义，再交给 minimax 在树中比较最终局面分数。
    plans: list[ActionPlan] = []
    for action_id, name, need_target in hero.get_available_actions():
        action_type = next(info[4] for info in hero.action_list if info[0] == action_id)
        target_index = enemy_index if need_target else None
        plans.append(ActionPlan(action_id, target_index, action_type, name))

    plans.sort(key=lambda plan: ACTION_ORDER.get(plan.action_type, 999))
    return plans


def _apply_round(heroes: Sequence[BasicHero], p1_plan: ActionPlan, p2_plan: ActionPlan) -> None:
    # 这里是 minimax 展开的核心单元：给定双方动作，模拟一个完整回合。
    p1, p2 = heroes
    actions = [(p1, p1_plan), (p2, p2_plan)]
    actions.sort(key=lambda item: ACTION_ORDER.get(item[1].action_type, 999))

    for hero in heroes:
        hero.reset_stacks()

    for hero, plan in actions:
        target = heroes[plan.target_index] if plan.target_index is not None else None
        hero.apply_action(plan.action_id, target, list(heroes))

    _clear_marked_actions(heroes)

    for hero in heroes:
        hero.update_status()


class MinimaxBattleAI:
    def __init__(self, genome: BattleGenome, search: SearchConfig | None = None):
        self.genome = genome
        self.search = search or SearchConfig()

    def evaluate(self, me: BasicHero, enemy: BasicHero) -> float:
        # 叶子节点的静态评估：把局面压缩成数值分数。
        return self.genome.score(BattleSnapshot.from_hero(me), BattleSnapshot.from_hero(enemy))

    def choose_action(self, me: BasicHero, enemy: BasicHero) -> ActionPlan:
        # 根节点决策：枚举己方所有候选动作，假设对手最优应对，
        # 选择最坏情况下仍然最优的那一个动作。
        actions = _legal_actions(me, 1)
        enemy_actions = _legal_actions(enemy, 0)

        if self.search.branch_limit > 0:
            actions = actions[: self.search.branch_limit]
            enemy_actions = enemy_actions[: self.search.branch_limit]

        best_plan = actions[0]
        best_value = -math.inf

        for my_plan in actions:
            worst_case = math.inf
            for enemy_plan in enemy_actions:
                next_heroes = deepcopy([me, enemy])
                _apply_round(next_heroes, my_plan, enemy_plan)
                value = self._minimax(next_heroes[0], next_heroes[1], self.search.depth - 1)
                worst_case = min(worst_case, value)

            if worst_case > best_value:
                best_value = worst_case
                best_plan = my_plan

        return best_plan

    def _minimax(self, me: BasicHero, enemy: BasicHero, depth: int) -> float:
        # 递归 minimax：
        # 1) 到深度上限或终局时，直接做静态评分；
        # 2) 否则对双方动作做双层枚举，己方取 max，对手取 min。
        winner = _winner_name([me, enemy])
        if depth <= 0 or winner is not None:
            score = self.evaluate(me, enemy)
            if winner == me.name:
                score += 10_000
            elif winner == enemy.name:
                score -= 10_000
            return score

        my_actions = _legal_actions(me, 1)
        enemy_actions = _legal_actions(enemy, 0)

        if self.search.branch_limit > 0:
            my_actions = my_actions[: self.search.branch_limit]
            enemy_actions = enemy_actions[: self.search.branch_limit]

        best_value = -math.inf
        for my_plan in my_actions:
            worst_value = math.inf
            for enemy_plan in enemy_actions:
                next_heroes = deepcopy([me, enemy])
                _apply_round(next_heroes, my_plan, enemy_plan)
                value = self._minimax(next_heroes[0], next_heroes[1], depth - 1)
                worst_value = min(worst_value, value)
            best_value = max(best_value, worst_value)
        return best_value


class GeneticTrainer:
    def __init__(self, config: GeneticConfig, search: SearchConfig | None = None, identity: TrainingIdentity | None = None):
        self.config = config
        self.search = search or SearchConfig(depth=1)
        self.identity = identity or TrainingIdentity()
        self.rng = random.Random(config.seed)

    def _initial_population(self) -> list[BattleGenome]:
        return [BattleGenome.random(self.rng) for _ in range(self.config.population_size)]

    def _tournament_pick(self, population: Sequence[BattleGenome], fitness: Sequence[float]) -> BattleGenome:
        contenders = self.rng.sample(list(range(len(population))), k=min(self.config.tournament_size, len(population)))
        winner_index = max(contenders, key=lambda idx: fitness[idx])
        return population[winner_index]

    def _make_heroes(self) -> tuple[BasicHero, BasicHero]:
        # 训练时使用固定角色名，这样导出的模型文件名也能和角色名对齐。
        return (
            BasicHero(self.identity.self_role, hp=5, hp_ceiling=5, mp=12, mp_ceiling=12),
            BasicHero(self.identity.enemy_role, hp=5, hp_ceiling=5, mp=12, mp_ceiling=12),
        )

    def _play_match(self, genome_a: BattleGenome, genome_b: BattleGenome) -> float:
        # 自博弈评估：两个 genome 互相对打，胜负和残局分数都纳入适应度。
        hero_a, hero_b = self._make_heroes()
        ai_a = MinimaxBattleAI(genome_a, self.search)
        ai_b = MinimaxBattleAI(genome_b, self.search)

        for _ in range(self.search.max_rounds):
            if _winner_name([hero_a, hero_b]) is not None:
                break

            plan_a = ai_a.choose_action(hero_a, hero_b)
            plan_b = ai_b.choose_action(hero_b, hero_a)
            _apply_round([hero_a, hero_b], plan_a, plan_b)

            if _winner_name([hero_a, hero_b]) is not None:
                break

        winner = _winner_name([hero_a, hero_b])
        if winner == hero_a.name:
            return 1000.0 + (hero_a.hp - hero_b.hp) * 10.0 + (hero_a.mp - hero_b.mp)
        if winner == hero_b.name:
            return -1000.0 + (hero_a.hp - hero_b.hp) * 10.0 + (hero_a.mp - hero_b.mp)
        return (hero_a.hp - hero_b.hp) * 10.0 + (hero_a.mp - hero_b.mp)

    def _fitness(self, genome: BattleGenome, population: Sequence[BattleGenome]) -> float:
        # 适应度：与种群中的若干对手对战，平均表现越高越好。
        if len(population) == 1:
            return 0.0

        opponents = [g for g in population if g is not genome]
        if not opponents:
            opponents = [genome]

        score = 0.0
        match_count = min(self.config.games_per_fitness, len(opponents))
        samples = self.rng.sample(opponents, k=match_count) if len(opponents) >= match_count else list(opponents)

        for opp in samples:
            score += self._play_match(genome, opp)
            score -= self._play_match(opp, genome) * 0.25

        return score / max(1, len(samples))

    def train(
        self,
        generations: Optional[int] = None,
        on_generation_end: Optional[Callable[[dict], None]] = None,
    ) -> tuple[BattleGenome, list[dict]]:
        # 遗传过程：
        # 1) 生成初代种群；
        # 2) 评估适应度并保留精英；
        # 3) 用锦标赛选择 + 交叉 + 变异生成下一代；
        # 4) 迭代到指定代数后返回最佳 genome。
        generations = generations or self.config.generations
        population = self._initial_population()
        history: list[dict] = []

        for generation in range(generations):
            fitness = [self._fitness(genome, population) for genome in population]
            ranked = sorted(zip(population, fitness), key=lambda item: item[1], reverse=True)
            elites_count = max(1, int(self.config.population_size * self.config.elite_rate))
            elites = [genome for genome, _ in ranked[:elites_count]]

            generation_record = {
                "generation": generation,
                "generation_index": generation + 1,
                "total_generations": generations,
                "best_fitness": ranked[0][1],
                "avg_fitness": sum(fitness) / len(fitness),
                "best_genome": ranked[0][0].to_dict(),
                "battle_name": self.identity.slug(),
            }
            history.append(generation_record)

            if on_generation_end is not None:
                on_generation_end(generation_record)

            next_population = list(elites)
            survivors = ranked[: max(2, int(self.config.population_size * (1.0 - self.config.elimination_rate)))]
            survivor_pool = [genome for genome, _ in survivors]

            while len(next_population) < self.config.population_size:
                pick_fitness = [self._fitness(g, survivor_pool) for g in survivor_pool]
                parent_a = self._tournament_pick(survivor_pool, pick_fitness)
                parent_b = self._tournament_pick(survivor_pool, pick_fitness)

                if self.rng.random() < self.config.crossover_rate:
                    child = BattleGenome.crossover(parent_a, parent_b, self.rng)
                else:
                    child = deepcopy(parent_a)

                child = child.mutated(self.rng, self.config.mutation_temperature, self.config.mutation_sigma)
                next_population.append(child)

            population = next_population

        final_fitness = [self._fitness(genome, population) for genome in population]
        best_index = max(range(len(population)), key=lambda idx: final_fitness[idx])
        return population[best_index], history

    def default_output_path(self, suffix: str = "json") -> Path:
        # 训练结果文件默认带上双方角色名，方便区分不同对局配置。
        return Path(f"best_genome_{self.identity.slug()}.{suffix}")

    def save_best(
        self,
        genome: BattleGenome,
        path: str | Path | None = None,
        history: Optional[Sequence[dict]] = None,
    ) -> Path:
        target = Path(path) if path is not None else self.default_output_path()
        payload = {
            "best_genome": genome.to_dict(),
        }

        if history:
            payload["training_summary"] = history[-1]

        # 按用户需求把训练设定写在输出文件末尾，便于回溯本次训练参数。
        payload["training_settings"] = {
            "identity": asdict(self.identity),
            "search": asdict(self.search),
            "genetic": asdict(self.config),
        }

        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def default_ai(depth: int = 2) -> MinimaxBattleAI:
    return MinimaxBattleAI(BattleGenome(), SearchConfig(depth=depth))
