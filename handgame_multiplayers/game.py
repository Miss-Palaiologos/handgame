from typing import Tuple, Optional, List
from ai import BattleGenome, MinimaxBattleAI, PlayerAIConfig, SearchConfig
from heros.basic import BasicHero
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import msvcrt
import os


class Game:
    """游戏主体类，管理流程与交互，具体结算由角色类完成。"""

    def __init__(self, *players: BasicHero, verbose: bool = False):
        # 支持任意数量玩家：传入多个 BasicHero 实例即可
        self.players = tuple(players)
        self.verbose = verbose
        self.round_no = 1
        # 每个位置都可以单独配置是否由 AI 托管，以及使用哪一组参数。
        self.player_ai_configs: list[Optional[PlayerAIConfig]] = [None for _ in self.players]
        self.player_ai_controllers: list[Optional[MinimaxBattleAI]] = [None for _ in self.players]
        # 可以提前注册多个命名配置，之后按名字切换。
        self.ai_profiles: dict[str, PlayerAIConfig] = {}
        self.ranking_list: dict[str,int] = {}  # 记录每局结束时玩家的排名
        self.console = Console()

    def register_ai_profile(
        self,
        profile_name: str,
        genome: Optional[BattleGenome] = None,
        search: Optional[SearchConfig] = None,
    ) -> None:
        """注册一组可复用的 AI 参数。"""
        self.ai_profiles[profile_name] = PlayerAIConfig(
            enabled=True,
            genome=genome or BattleGenome(),
            search=search or SearchConfig(),
            label=profile_name,
        )

    def set_player_ai(
        self,
        player_index: int,
        enabled: bool = True,
        profile_name: Optional[str] = None,
        genome: Optional[BattleGenome] = None,
        search: Optional[SearchConfig] = None,
        label: Optional[str] = None,
    ) -> None:
        """给某个玩家位置设置是否托管，以及使用哪一组参数。"""
        if not 0 <= player_index < len(self.players):
            raise IndexError("player_index must be 0 or 1")

        if profile_name is not None:
            profile = self.ai_profiles.get(profile_name)
            if profile is None:
                raise KeyError(f"未知的 AI 配置: {profile_name}")
            config = profile
        else:
            config = PlayerAIConfig(
                enabled=enabled,
                genome=genome or BattleGenome(),
                search=search or SearchConfig(),
                label=label or f"player{player_index + 1}",
            )

        self.player_ai_configs[player_index] = config if enabled else None
        self.player_ai_controllers[player_index] = (
            MinimaxBattleAI(config.genome, config.search) if enabled else None
        )

    def set_player_ai_profile(self, player_index: int, profile_name: str) -> None:
        """快速把某个玩家切换到已注册的命名 AI 配置。"""
        self.set_player_ai(player_index, enabled=True, profile_name=profile_name)

    def is_player_ai(self, player_index: int) -> bool:
        """检查某个玩家是否由 AI 托管。"""
        return self.player_ai_configs[player_index] is not None

    def debug_print(self, message: str) -> None:
        if self.verbose:
            print(message)

    def show_status(self) -> None:
        print("当前状态：")
        for player in self.players:
            print(f"  {player.get_status_display()}")

    def prompt_action(self, player_index: int, player: BasicHero) -> Tuple[int, Optional[int]]:
        while True:
            available_actions = player.get_available_actions()
            if not available_actions:
                self.console.print("没有可用行动！")
                return 0, None

            action_line = "  ".join(
                f"{action_id}. {name} [MP {next((info[2] for info in player.action_list if info[0] == action_id), 0):+d}]"
                + (" (需要目标)" if need_target else "")
                for action_id, name, need_target in available_actions
            )

            raw = ""

            def generate_panel(raw: str) -> Panel:
                grid = Table.grid(expand=False)
                grid.add_row(f"{player.name} 请选择行动（当前MP {player.mp}/{player.mp_ceiling}，HP {player.hp}/{player.hp_ceiling}）：")
                grid.add_row(action_line)
                grid.add_row("输入行动编号：" + raw)

                panel = Panel(grid, title=f"玩家 {player.name} 输入")
                return panel
                

            with Live(generate_panel(raw), console=self.console, transient=True) as live:
                while True:
                    if msvcrt.kbhit():  # 检查是否有按键按下
                        ch = msvcrt.getch().decode("utf-8", errors="ignore")  # 读取按键并解码
                        
                        # 处理回车 (Windows 下回车通常是 \r 或 \n)
                        if ch in ('\r', '\n'):
                            break
                        
                        # 处理退格 (Backspace)
                        elif ch == '\x08':
                            raw = raw[:-1]

                        # 处理其他可打印字符
                        elif ch.isprintable():
                            raw += ch
                        
                        # 更新面板显示
                        live.update(generate_panel(raw))

            try:
                action_id = int(raw)
            except Exception:
                self.console.print("请输入有效的整数编号。")
                continue

            if not any(aid == action_id for aid, _, _ in available_actions):
                self.console.print(f"行动 {action_id} 不可用。")
                continue

            _, _, need_target = next((a for a in available_actions if a[0] == action_id), (None, None, False))

            target_index = None
            if need_target:
                target_index = self.prompt_target(player_index, player)
                if target_index is None:
                    continue

            return action_id, target_index

    def prompt_target(self, player_index: int, player: BasicHero) -> Optional[int]:
        # 将当前玩家索引传入角色的目标生成器，角色可以据此返回合法目标
        available_targets = player.get_available_target(player_index, list(self.players))

        while True:
            target_line = "  ".join(f"{target_id}. {target_name}" for target_id, target_name in available_targets)
            grid = Table.grid(expand=False)
            grid.add_row(f"{player.name} 请选择目标（输入 back 返回选择行动）：")
            grid.add_row(target_line)

            panel = Panel(grid, title=f"{player.name} 目标选择")
            with Live(panel, console=self.console, transient=True):
                raw = self.console.input("输入目标编号或 back：").strip().lower()

            if raw == "back":
                return None

            try:
                target_id = int(raw)
                if any(tid == target_id for tid, _ in available_targets):
                    return target_id
                self.console.print(f"目标 {target_id} 不可用。")
            except ValueError:
                self.console.print("请输入有效的整数编号或 back。")

    def action_selection_phase(self) -> Tuple[Tuple[int, Optional[int]], ...]:
        # 按玩家注册顺序逐一询问行动，询问完成后清屏，避免暴露前一位玩家的选择
        plans: list[Tuple[int, Optional[int]]] = []
        total = len(self.players)
        for idx, player in enumerate(self.players):
            plan = self._choose_action_for_player(idx, player)
            plans.append(plan)

        return tuple(plans)

    def _choose_action_for_player(self, player_index: int, player: BasicHero) -> Tuple[int, Optional[int]]:
        """多玩家版本的出招选择；AI 会以下一个玩家作为默认对手用于决策。"""
        controller = self.player_ai_controllers[player_index]
        if controller is not None:
            # 选择下一个索引作为默认敌手（回合制多人大多数场景以邻位为目标）
            enemy_index = (player_index + 1) % len(self.players)
            enemy = self.players[enemy_index]
            plan = controller.choose_action(player, enemy)
            self.debug_print(
                f"{player.name} (AI) 选择: {plan.action_id} / {plan.action_name} / target={plan.target_index}"
            )
            return plan.action_id, plan.target_index

        return self.prompt_action(player_index, player)

    def action_display_phase(self, actions) -> None:
        # actions: 可为 list/tuple of (action_id, target_idx)
        actions_list = list(actions)

        print("\n========== 行动展示阶段 ==========")
        for idx, (action_id, target_idx) in enumerate(actions_list):
            player = self.players[idx]
            name = next((info[1] for info in player.action_list if info[0] == action_id), "未知")
            target_info = ""
            if target_idx is not None and 0 <= target_idx < len(self.players):
                target_info = f" → {self.players[target_idx].name}"
            print(f"{player.name}: {action_id} - {name}{target_info}")

    def apply_actions_phase(self, actions) -> None:
        # actions: iterable of (action_id, target_idx) 对应 players 顺序
        actions_list = list(actions)

        all_heroes = list(self.players)

        action_order = {
            "self-attack": 0,
            "group-attack": 1,
            "anti-group-attack": 2,
            "attack": 3,
            "defense": 4,
            "control": 5,
            None: 6,
        }

        work_list: List[tuple[BasicHero, int, Optional[int], Optional[str]]] = []
        for idx, (action_id, target_idx) in enumerate(actions_list):
            player = self.players[idx]
            info = next((info for info in player.action_list if info[0] == action_id), None)
            action_type = info[4] if info is not None else None
            work_list.append((player, action_id, target_idx, action_type))

        work_list.sort(key=lambda x: action_order.get(x[3], 999))

        for player, action_id, target_idx, action_type in work_list:
            target = all_heroes[target_idx] if target_idx is not None and 0 <= target_idx < len(all_heroes) else None
            self.debug_print(f"{player.name} 应用行动 {action_id}")
            player.apply_action(action_id, target, all_heroes)

        # 统一清理被标记的自戕攻击（支持多人反弹）
        for hero in all_heroes:
            for action in hero.attack_stack:
                if getattr(action, "marked_for_clear", False):
                    action.attack_value = 0
                    action.marked_for_clear = False
            for action in hero.defense_stack:
                if getattr(action, "marked_for_clear", False):
                    action.attack_value = 0
                    action.marked_for_clear = False

    def update_status_phase(self) -> None:
        for player in self.players:
            player.update_status()

    def mark_phase(self) -> None:
        self.debug_print("\n========== 印记结算阶段 ==========")
        self.debug_print("当前版本未实现额外印记效果。")
        for player in self.players:
            player.apply_stamp()

    def death_phase(self) -> None:
        self.debug_print("\n========== 死亡结算阶段 ==========")
        for player in self.players:
            if player.is_defeated():
                print(f"{player.name} 已死亡。")

    def resolve_round(self) -> None:
        print()
        print(f"========== 第 {self.round_no} 回合 ==========")
        self.show_status()

        for player in self.players:
            player.reset_stacks()

        actions = self.action_selection_phase()
        self.action_display_phase(actions)
        self.apply_actions_phase(actions)
        self.update_status_phase()
        self.mark_phase()
        self.update_status_phase()
        self.death_phase()

    def get_winner(self) -> Optional[str]:
        alive = [player for player in self.players if not player.is_defeated()]

        for player in self.players:
            if player.is_defeated() and player.name not in self.ranking_list.keys():
                self.ranking_list[player.name] = len(alive)+1

        if len(alive) == 1 or (len(alive) == 0 and len(self.ranking_list) == len(self.players)):
            return self.ranking_list
        return None

    def run(self) -> None:
        while True:
            self.resolve_round()
            result = self.get_winner()
            if result is not None:
                print("\n========== 对局结束 ==========")
                print(f"player排名: {result}")
                break

            self.round_no += 1
            print()
            input("按回车进入下一回合。")
