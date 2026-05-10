from typing import List, Tuple, Optional


class attack_action:
    def __init__(self, attack_value, target, source, attack_type, damage_type):
        self.attack_value = attack_value
        self.target = target
        self.source = source
        self.attack_type = attack_type
        self.damage_type = damage_type
        self.marked_for_clear = False


class BasicHero:
    """基础角色类，使用堆栈机制管理行动和效果。"""

    def __init__(
        self,
        name: str,
        hp: int = 5,
        hp_ceiling: int = 5,
        mp: int = 0,
        mp_ceiling: int = 12,
        damage_resistance: int = 0,
        shield: int = 0,
        revival_armor: int = 0,
    ):
        self.name = name
        self.hp = hp
        self.hp_ceiling = hp_ceiling
        self.mp = mp
        self.mp_ceiling = mp_ceiling
        self.damage_resistance = damage_resistance
        self.shield = shield
        self.shield_ceiling = shield
        self.revival_armor = revival_armor

        self.attack_stack: List[attack_action] = []
        self.defense_stack: List[attack_action] = []
        self.mp_stack: List[int] = []
        self.hp_stack: List[int] = []
        self.hp_ceiling_stack: List[int] = []
        self.stamp_stack: List[Tuple[int, int, str, Optional[str]]] = []
        self.action = None

        self.action_list = [
            (0, "攒能", 0, False, None, 0, None),
            (1, "小防", 0, False, "defense", 0, None),
            (2, "大防", 1, False, "defense", 0, None),
            (3, "反弹", 1, False, "control", 0, None),
            (4, "清零", 2, False, "control", 0, None),
            (5, "护盾", 3, False, "control", 0, None),
            (6, "自戕", 0, False, "self-attack", 2, "real"),
            (7, "单枪", 1, True, "attack", 1, "physical"),
            (8, "双枪", 2, True, "anti-group-attack", 2, "physical"),
            (9, "大招", 3, False, "group-attack", 3, "physical"),
            (10, "复活甲", 6, False, None, 0, None),
        ]

    def reset_stacks(self):
        """重置回合堆栈，在每回合开始时调用。"""
        self.attack_stack.clear()
        self.defense_stack.clear()
        self.mp_stack.clear()
        self.hp_stack.clear()
        self.hp_ceiling_stack.clear()
        self.damage_resistance = 0

    def _append_shared_attack(self, action: attack_action, target: "BasicHero") -> None:
        self.attack_stack.append(action)
        target.defense_stack.append(action)

    def _zero_actions(self, actions: List[attack_action]) -> None:
        for action in actions:
            action.attack_value = 0

    def _sum_actions(self, actions: List[attack_action], allowed_types: Optional[List[str]] = None) -> int:
        if allowed_types is None:
            return sum(a.attack_value for a in actions)
        return sum(a.attack_value for a in actions if a.attack_type in allowed_types)

    def get_available_actions(self):
        available = []
        for action_id, name, cost, need_target, *_ in self.action_list:
            if self.mp >= cost:
                available.append((action_id, name, need_target))
        return available

    def get_available_target(self, action_id, player_list):
        available_targets = []
        for i, player in enumerate(player_list):
            if player != self:
                available_targets.append((i, player.name))
        return available_targets

    def apply_action(self, action_id, target=None, all_heroes: Optional[List["BasicHero"]] = None):
        action_info = None
        for info in self.action_list:
            if info[0] == action_id:
                action_info = info
                break

        if not action_info:
            return

        action_id, name, cost, need_target, action_type, attack_value, damage_type = action_info
        self.mp_stack.append(-cost)

        if action_id == 0:
            self.mp_stack.append(1)
            return

        if action_type == "self-attack":
            action = attack_action(attack_value, self, self, "self-attack", damage_type)
            self._append_shared_attack(action, self)
            return

        if action_type == "group-attack":
            if all_heroes:
                for hero in all_heroes:
                    if hero != self:
                        action = attack_action(attack_value, hero, self, "group-attack", damage_type)
                        self._append_shared_attack(action, hero)
            return

        if action_type == "anti-group-attack":
            if target:
                self._apply_anti_group_attack(target, attack_value, damage_type)
            return

        if action_type == "attack":
            if target:
                self._apply_attack(target, attack_value, damage_type)
            return

        if action_type == "defense":
            if action_id == 1:
                self._apply_small_defense(2)
            elif action_id == 2:
                self._apply_large_defense(6)
            return

        if action_type == "control":
            if action_id == 3:
                self._apply_ricochet(all_heroes)
            elif action_id == 4:
                self._apply_clear_zero(all_heroes)
            elif action_id == 5:
                self.shield += 1
            return

        if action_id == 10:
            self._apply_revive_armor(all_heroes)

    def _apply_attack(self, target, attack_value, damage_type):
        action = attack_action(attack_value, target, self, "attack", damage_type)
        self._append_shared_attack(action, target)

        has_group_attack = any(
            a.attack_type == "group-attack" and a.source == target
            for a in self.defense_stack
        )
        if has_group_attack:
            action.attack_value = 0
            return

        counter = next(
            (
                a for a in self.defense_stack
                if a.attack_type == "anti-group-attack" and a.source == target
            ),
            None,
        )
        if counter is not None:
            self._resolve_attack_diff(action, counter)

    def _apply_anti_group_attack(self, target, attack_value, damage_type):
        action = attack_action(attack_value, target, self, "anti-group-attack", damage_type)
        self._append_shared_attack(action, target)

        group_actions = [
            a for a in self.defense_stack
            if a.attack_type == "group-attack" and a.source == target
        ]
        if group_actions:
            self._zero_actions(group_actions)
            return

        counter = next(
            (
                a for a in self.defense_stack
                if a.attack_type == "anti-group-attack" and a.source == target
            ),
            None,
        )
        if counter is not None:
            self._resolve_attack_diff(action, counter)

    def _resolve_attack_diff(self, action: attack_action, counter: attack_action) -> None:
        new_action = max(0, action.attack_value - counter.attack_value)
        new_counter = max(0, counter.attack_value - action.attack_value)
        action.attack_value = new_action
        counter.attack_value = new_counter

    def _apply_small_defense(self, defense_value):
        target_actions = [
            a for a in self.defense_stack
            if a.attack_type in ("attack", "anti-group-attack")
        ]
        total = self._sum_actions(target_actions)
        if total <= defense_value:
            self._zero_actions(target_actions)
        else:
            self.damage_resistance += defense_value

    def _apply_large_defense(self, defense_value):
        target_actions = [
            a for a in self.defense_stack
            if a.attack_type in ("attack", "anti-group-attack")
        ]
        total = self._sum_actions(target_actions)
        if total <= defense_value:
            self._zero_actions(target_actions)
        else:
            self.damage_resistance += defense_value

        group_actions = [a for a in self.defense_stack if a.attack_type == "group-attack"]
        self._zero_actions(group_actions)

    def _apply_ricochet(self, all_heroes: Optional[List["BasicHero"]] = None) -> None:
        if not all_heroes:
            return

        triggered = False
        for hero in all_heroes:
            for action in hero.attack_stack:
                if action.attack_type == "self-attack" and action.attack_value > 0:
                    reflected = attack_action(
                        action.attack_value,
                        self,
                        action.source,
                        "attack",
                        "physical",
                    )
                    action.source.attack_stack.append(reflected)
                    self.defense_stack.append(reflected)
                    # 标记原始自戕对象，延迟在控制阶段结束后统一清除
                    action.marked_for_clear = True
                    triggered = True

        if triggered:
            return

        target_actions = [
            a for a in self.defense_stack
            if a.attack_type in ("attack", "anti-group-attack")
        ]
        total = self._sum_actions(target_actions)
        if total <= 6:
            for action in target_actions:
                if action in self.defense_stack:
                    self.defense_stack.remove(action)
                action.target = action.source
                action.source.defense_stack.append(action)

    def _apply_clear_zero(self, all_heroes: Optional[List["BasicHero"]] = None) -> None:
        total = self._sum_actions(self.defense_stack)
        if total >= 3 or total == 0:
            self._zero_actions(self.defense_stack)
            self.hp_stack.append(1)
            if all_heroes:
                for hero in all_heroes:
                    hero.mp_stack.append(-999)
        else:
            self.hp_stack.append(-1)

    def _apply_revive_armor(self, all_heroes: Optional[List["BasicHero"]] = None) -> None:
        self._zero_actions(self.defense_stack)
        self.revival_armor += 1
        if all_heroes:
            for hero in all_heroes:
                hero.mp_stack.append(-999)

    def update_status(self):
        if self.shield > 0:
            has_blockable = any(
                a.attack_value > 0 and a.damage_type in ("physical", "magic")
                for a in self.defense_stack
            )
            if has_blockable:
                for action in self.defense_stack:
                    if action.damage_type in ("physical", "magic"):
                        action.attack_value = 0
                self.shield -= 1

        physical_damage = 0
        real_damage = 0
        magic_damage = 0
        for action in self.defense_stack:
            if action.attack_value <= 0:
                continue
            amount = action.attack_value
            if action.attack_type == "self-attack":
                amount = amount // 2
            if action.damage_type == "physical":
                physical_damage += amount
            elif action.damage_type == "real":
                real_damage += amount
            elif action.damage_type == "magic":
                magic_damage += amount

        if physical_damage:
            # 防御超上限只影响结算结果，不修改标签
            reduced = max(0, physical_damage - self.damage_resistance)
            if reduced:
                self.hp_stack.append(-reduced)
        if real_damage:
            self.hp_stack.append(-real_damage)
        if magic_damage:
            self.hp_ceiling_stack.append(-magic_damage)

        self.hp += sum(self.hp_stack)
        self.hp_ceiling += sum(self.hp_ceiling_stack)
        if self.hp > self.hp_ceiling:
            self.hp = self.hp_ceiling

        self.mp += sum(self.mp_stack)
        self.mp = max(0, min(self.mp, self.mp_ceiling))

        self.hp_stack.clear()
        self.hp_ceiling_stack.clear()
        self.mp_stack.clear()

    def apply_stamp(self):
        pass

    def is_defeated(self):
        if self.hp <= 0 and self.revival_armor > 0:
            self.revival_armor -= 1
            self.hp = 1
            self.mp = 1
            self.shield = 0
            self.damage_resistance = 0
            self.attack_stack.clear()
            self.defense_stack.clear()
            self.mp_stack.clear()
            self.hp_stack.clear()
            self.hp_ceiling_stack.clear()
            self.stamp_stack.clear()
            return False
        return self.hp <= 0

    def get_status_display(self) -> str:
        hp_blocks = "■" * max(0, self.hp) if self.hp > 0 else ""
        mp_blocks = "■" * max(0, self.mp) if self.mp > 0 else ""
        return (
            f"{self.name}: HP {hp_blocks or '空'} ({self.hp}/{self.hp_ceiling}), "
            f"MP {mp_blocks or '空'} ({self.mp}/{self.mp_ceiling}), 复活甲 {self.revival_armor}"
        )
