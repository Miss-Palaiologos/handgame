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

class hero:
    def __init__(self, name, hp=5, hp_ceiling=5, mp=0, mp_ceiling=12,damage_resistance=0, shield=0, revival_armor=0):
        self.name = name
        self.hp = hp
        self.hp_ceiling = hp_ceiling
        self.mp = mp
        self.mp_ceiling = mp_ceiling
        self.damage_resistance = damage_resistance
        self.shield = shield
        self.shield_ceiling = shield
        self.revival_armor = revival_armor

        self.attack_stack = []  # 攻击堆栈，记录当前回合的攻击行为
        self.defense_stack = []  # 防御堆栈，记录当前回合的防御行为
        self.mp_stack = []  # 能量堆栈，记录当前回合的能量变化(mp_change)
        self.hp_stack = []  # 生命堆栈，记录当前回合的生命变化(hp_change)
        self.hp_ceiling_stack = []  # 生命上限堆栈，记录当前回合的生命上限变化(hp_ceiling_change)
        self.stamp_stack = []  # 印记堆栈，暂时先不设计具体内容，预留给后续设计使用
        self.action = None  # 当前选择的行动ID

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
        不取对象：将自己从本回合的全部结算中移除
        自戕：直接把相关攻击写入自己的攻击堆栈和防御堆栈（自戕攻击既算攻击又算防御，且不受任何防御类行动影响）
        群体攻击：检查自己的防御堆栈中是否有目标的群体攻击，如果有则基于当前快照同时分别与全部群体攻击的attack_value相减并与0取max重新写回彼此的攻击与防御堆栈，否则直接把攻击写入目标的防御堆栈和自己的攻击堆栈
        反群体攻击：检查自己的防御堆栈中是否有目标的群体攻击，如果有则把所有群体攻击从自己的防御堆栈和其对应的攻击堆栈中的attack_value设为0，并把反群体攻击添加到自己的攻击堆栈和目标的防御堆栈
                   检查自己的防御堆栈中是否有目标的反群体攻击，如果有则把双方的攻击数值相减并与0取max重新写回彼此的攻击与防御堆栈
                   如果上述两种情况都不存在，则直接把相关攻击写入目标的攻击堆栈和自己的防御堆栈
        攻击：检查自己的防御堆栈中是否有目标的群体攻击，如果有则什么都不做
             检查自己的防御堆栈中是否有目标的反群体攻击，如果有则把双方的攻击数值相减并与0取max重新写回彼此的攻击与防御堆栈
             如果上述两种情况都不存在，则直接把相关攻击写入目标的攻击堆栈和自己的防御堆栈
        防御：小防：检查自己的防御堆栈中的反群体攻击和攻击的数值总和，若总和小于等于自己的防御数值2，则把相关攻击的在自己的防御堆栈和攻击方的攻击堆栈中的attack_value设为0
                   否则令自己的damage_resistance加2
              大防：检查自己的防御堆栈中的反群体攻击和攻击的数值总和，若总和小于等于自己的防御数值6，则把相关攻击的在自己的防御堆栈和攻击方的攻击堆栈中的attack_value设为0
                   否则令自己的damage_resistance加6
                   检查自己的防御堆栈中是否有群体攻击，如果有则把相关攻击的在自己的防御堆栈和攻击方的攻击堆栈中的attack_value设为0（群体攻击永远只移除对自己的那一份）
        控制：反弹：检查场上是否有自戕标签的攻击，如果有，则将与其攻击堆栈中attack_value相同的攻击用普通attack标签写入自戕发出者的攻击堆栈和自己的防御堆栈，在本阶段结束时，只要有反弹触发了这个机制，则清除自戕者两个堆栈中的自戕攻击本身
                                              如果没有，检查自己的防御堆栈中的反群体攻击和攻击的数值总和，若总和小于等于自己的防御数值6，则把相关攻击从自己的防御堆栈中移入攻击方的防御堆栈，并修改攻击方的防御堆栈与攻击堆栈中的目标
             清零：检查自己的防御堆栈中所有种类攻击的数值总和，若总和大于等于3或等于0，则把相关攻击的在自己的防御堆栈和攻击方的攻击堆栈中的attack_value设为0，给自己在hp_stack中添加1的血量变化，并给全场所有人的mp_stack中添加一个-999的能量变化
                   否则给自己的hp_stack中添加-1的血量变化
             护盾：给自己的护盾值加1
        无标记：攒能：给自己的mp_stack中添加1的能量变化
               复活甲：移除自己的防御堆栈内全部攻击和对应攻击者攻击堆栈内的对应攻击，给自己的复活甲数目+1，给所有玩家的mp_stack中添加一个-999的能量变化
        '''
        pass

    def update_status(self):
        # 若自己的护盾不为0，则检查自己的防御栈内是否存在"physical"或"magic"类型的攻击，若有，则将全部这些攻击与对应攻击者的攻击栈中的attack_value设为0，并让shield值-1
        # 否则直接按类别求和，"physical"属性攻击数值在求和后减去damage_resistance的数值加入到hp_stack中，real的攻击直接减少血量加入到hp_stack中（注意标注为self-attack的attack_value折半计入伤害），magic的攻击减少血量上限加入到hp_ceiling_stack中
        # 在每回合行动应用阶段结束和印记结算阶段时调用，根据hp_stack和mp_stack更新玩家的生命和能量，将负值设定为0，并清空hp_stack和mp_stack
        pass

    def apply_stamp(self):
        # 在印记结算阶段调用，循环弹出自己的stamp_stack中的印记，根据印记的source调用对应角色的stamp_effect函数来应用印记效果，印记效果可能会基于印记的name和source来对target造成伤害、治疗、增益、减益等
        pass

    def stamp_effect(self,stamp: stamp):
        # 在其他玩家的apply_stamp函数中被调用，根据stamp的name和source来对target造成伤害、治疗、增益、减益等
        pass

    def is_defeated(self):
        # 判断玩家是否被击败，条件是生命小于等于0且没有复活甲，返回布尔值
        # 如果有复活甲，复活甲数量-1并直接重设hp=1，mp=1，shield=0，damage_resistance=0，清空全部堆栈
        pass