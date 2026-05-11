import basic

class stamp:
    def __init__(self, source, target, name):
        self.source = source  # 印记来源
        self.target = target  # 印记目标 
        self.name = name

class attack_action:
    def __init__(self, attack_value, target, source, attack_type, damage_type):
        self.attack_value = attack_value
        self.target = target  # 目标玩家对象
        self.source = source  # 攻击者对象
        self.attack_type = attack_type  # 攻击类型，如 self-attack、group-attack 等
        self.damage_type = damage_type  # 伤害类型，如 physical、magic 等

class Hunter(basic.BasicHero):
    def __init__(self, name, hp=6, hp_ceiling=6, mp=0, mp_ceiling=12,damage_resistance=0, shield=0, revival_armor=0):
        super().__init__(name, hp, hp_ceiling, mp, mp_ceiling, damage_resistance, shield, revival_armor)

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
        ]   #所有行动及其信息列表，格式为[(action_id, action_name, action_cost, if_need_target, action_type, attack_value, damage_type), ...]

    def reset_stacks(self):
        """重置回合堆栈，在每回合开始时调用。"""
        self.attack_stack.clear()
        self.defense_stack.clear()
        self.mp_stack.clear()
        self.hp_stack.clear()
        self.damage_resistance = 0

    def get_available_actions(self):
        #根据当前的能量返回可用的行动列表，格式为[(action_id, action_name, if_need_target), ...]
        pass
    
    def get_available_target(self, action_id, player_list):
        # 如果行动需要选择目标，返回可选目标列表，格式为[(target_id, target_name), ...]
        # 此时可以按backspace返回上一步选择
        pass

    def apply_action(self, action_id, target=None):
        # 在玩家全部选择完行动之后，主函数将每个玩家的行为按照行为类型进行分类
        # 遵循自戕-群体攻击-反群体攻击-攻击-防御-控制-无标记的顺序调用此函数来应用玩家的行动效果
        # 默认把能量消耗写入mp_stack，下述不反复提及
        '''
        攻击：单枪：改为不需要和非猎人的反群体攻击和攻击进行数值差分，直接把相关攻击写入目标的防御堆栈和自己的攻击堆栈，并将对方的反群体攻击和攻击的attack_value设为0
             三枪：与双枪机制相同，但是目标使用了群体攻击触发反群体机制时造成1点伤害而非3点
                   并检查自己的防御堆栈中全部的群体攻击，给这些群体攻击的来源stamp“反击”
        空城：在每回合的update_status阶段，对自己的防御堆栈进行检查，对全部的除自己的“physical”属性攻击来源附加stamp“空城”

        '''
        pass

    def update_status(self):
        # 若自己的护盾不为0，则检查自己的防御栈内是否存在"physical"或"magic"类型的攻击，若有，则将全部这些攻击与对应攻击者的攻击栈中的attack_value设为0，并让shield值-1
        # 否则直接按类别求和，"physical"属性攻击数值在求和后减去damage_resistance的数值加入到hp_stack中，real的攻击直接减少血量加入到hp_stack中（注意标注为self-attack的attack_value折半计入伤害），magic的攻击减少血量上限加入到hp_ceiling_stack中
        # 记录这回合自己的受到的伤害是不是0
        # 在每回合行动应用阶段结束和印记结算阶段时调用，根据hp_stack和mp_stack更新玩家的生命和能量，将负值设定为0，并清空hp_stack和mp_stack
        # 记录这回合行动结算结束自己的能量是不是0
        pass

    def apply_stamp(self):
        # 在印记结算阶段调用，循环弹出自己的stamp_stack中的印记，根据印记的source调用对应角色的stamp_effect函数来应用印记效果，印记效果可能会基于印记的name和source来对target造成伤害、治疗、增益、减益等
        pass

    def stamp_effect(self,stamp: stamp):
        # 在其他玩家的apply_stamp函数中被调用，根据stamp的name和source来对target造成伤害、治疗、增益、减益等
        # “反击”标签给予目标1点无来源，类型为无的真实伤害
        # “空城”标签首先判断自己的能量是不是0，如果是则给予目标1点无来源无类型的真实伤害，再检查自己是不是没有受到伤害，如果没受到伤害则给予目标额外的1点无来源无类型的真实伤害
        pass

    def is_defeated(self):
        # 判断玩家是否被击败，条件是生命小于等于0且没有复活甲，返回布尔值
        # 如果有复活甲，复活甲数量-1并直接重设hp=1，mp=1，shield=0，damage_resistance=0，清空全部堆栈
        pass