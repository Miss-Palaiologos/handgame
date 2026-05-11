from typing import List, Optional

from heros.basic import BasicHero, attack_action, stamp


class Hunter(BasicHero):
    """猎人职业：在基础角色之上扩展单枪、三枪与印记机制。"""

    def __init__(
        self,
        name: str,
        hp: int = 6,
        hp_ceiling: int = 6,
        mp: int = 0,
        mp_ceiling: int = 12,
        damage_resistance: int = 0,
        shield: int = 0,
        revival_armor: int = 0,
    ):
        super().__init__(name, hp, hp_ceiling, mp, mp_ceiling, damage_resistance, shield, revival_armor)
        self._mp_zero_last_settlement = False
        self._no_damage_last_settlement = False

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
            (9, "三枪", 2, True, "anti-group-attack", 3, "physical"),
            (10, "大招", 3, False, "group-attack", 3, "physical"),
            (11, "复活甲", 6, False, None, 0, None),
        ]

    def apply_action(self, action_id, target=None, all_heroes: Optional[List["BasicHero"]] = None):
        action_info = None
        for info in self.action_list:
            if info[0] == action_id:
                action_info = info
                break

        if not action_info:
            return

        action_id, _, cost, _, action_type, attack_value, damage_type = action_info
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
                        self._apply_group_attack(hero, attack_value, damage_type)
            return

        if action_type == "anti-group-attack":
            if not target:
                return
            if action_id == 9:
                self._apply_triple_shot(target, attack_value, damage_type)
            else:
                self._apply_anti_group_attack(target, attack_value, damage_type)
            return

        if action_type == "attack":
            if target:
                self._apply_hunter_single_shot(target, attack_value, damage_type)
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

        if action_id == 11:
            self._apply_revive_armor(all_heroes)

    def _apply_hunter_single_shot(self, target: BasicHero, attack_value: int, damage_type: str) -> None:
        """单枪对非猎人不做攻防差分，并清空对手的 attack/anti-group-attack。"""
        action = attack_action(attack_value, target, self, "attack", damage_type)
        self._append_shared_attack(action, target)

        has_group_attack = any(
            a.attack_type == "group-attack" and a.source == target
            for a in self.defense_stack
        )
        if has_group_attack:
            action.attack_value = 0
            return

        if isinstance(target, Hunter):
            counter = next(
                (
                    a for a in self.defense_stack
                    if a.attack_type == "anti-group-attack" and a.source == target
                ),
                None,
            )
            if counter is not None:
                self._resolve_attack_diff(action, counter)
            return

        for defense_action in target.defense_stack:
            if defense_action.attack_type in ("attack", "anti-group-attack"):
                defense_action.attack_value = 0

    def _apply_triple_shot(self, target: BasicHero, attack_value: int, damage_type: str) -> None:
        """三枪：双枪机制，若触发反群体则伤害改为 1，并给群体攻击来源添加反击印记。"""
        action = attack_action(attack_value, target, self, "anti-group-attack", damage_type)
        self._append_shared_attack(action, target)

        group_actions = [a for a in self.defense_stack if a.attack_type == "group-attack"]
        if group_actions:
            for group_action in group_actions:
                group_action.source.stamp_stack.append(stamp(self, group_action.source, "反击"))
            self._zero_actions(group_actions)
            action.attack_value = 1
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

    def update_status(self):
        # 空城印记在结算前根据当前回合来源打标，避免被 super().update_status() 消耗 defense_stack 后丢失来源信息。
        stamped_sources = set()
        for action in self.defense_stack:
            if action.damage_type != "physical":
                continue
            if action.source == self:
                continue
            source_id = id(action.source)
            if source_id in stamped_sources:
                continue
            action.source.stamp_stack.append(stamp(self, action.source, "空城"))
            stamped_sources.add(source_id)

        hp_before = self.hp
        super().update_status()
        self._no_damage_last_settlement = self.hp >= hp_before
        self._mp_zero_last_settlement = self.mp == 0

    def apply_stamp(self):
        while self.stamp_stack:
            current_stamp = self.stamp_stack.pop(0)
            current_stamp.source.stamp_effect(current_stamp)

    def stamp_effect(self, current_stamp: stamp):
        if current_stamp.name == "反击":
            current_stamp.target.hp_stack.append(-1)
            return

        if current_stamp.name == "空城":
            if self._mp_zero_last_settlement:
                current_stamp.target.hp_stack.append(-1)
                if self._no_damage_last_settlement:
                    current_stamp.target.hp_stack.append(-1)

