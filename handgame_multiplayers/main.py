from game import Game
from heros.basic import BasicHero

def main() -> None:
    print("双人控制台对战：手牌小游戏")
    print("按提示选择行动，两个玩家轮流在同一终端操作。")
    print()

    while True:
        verbose_choice = input("是否启用详细输出模式？(y/n，默认n): ").strip().lower()
        if verbose_choice in ["y", "yes"]:
            verbose = True
            print("已启用详细输出模式。")
            break
        if verbose_choice in ["n", "no", ""]:
            verbose = False
            print("已禁用详细输出模式。")
            break
        print("请输入 y 或 n。")

    p1 = BasicHero("玩家1")
    p2 = BasicHero("玩家2")
    p3 = BasicHero("玩家3")
    p4 = BasicHero("玩家4")
    game = Game(p1, p2, p3, p4, verbose=verbose)
    game.run()


if __name__ == "__main__":
    main()
