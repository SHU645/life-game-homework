# -*- coding: utf-8 -*-
"""
生命游戏 · 爱心作图【稳定版】
================================================================

【和上一版的区别】
    上一版 life_heart.py：爱心只在第 1 步出现，之后就开始消散。
    这一版           ：爱心一出现就**永远停在那里**（数学家管这叫"静物"）。

【为什么"稳定"难得多】
    生命游戏里绝大多数图案都不稳定——演化几步就散架。
    要让爱心永远不变，必须满足 **静物的严格条件**：

        对每一个格子（看它周围 8 个邻居）：
            · 活细胞  →  邻居数必须**恰好是 2 或 3**
            · 死细胞  →  邻居数**不能是 3**（否则会被"生"出来）

    任何一格不满足，整个图案就会变。所以这不是"画得像"就行，
    而是一个**约束满足问题**。

【怎么设计一个稳定的爱心：两个关键发现】

    发现一：2×2 的实心方块本身是静物
        验证：
            ##
            ##
        每个格子都恰好有 3 个邻居 → 稳定。这是生命游戏里最小的静物。

    发现二：方块之间必须留 **2 格空隙**（我踩过的坑）
        一开始我以为留 1 格就够（间距 3），结果有 12 个格子会变。
        原因：间距太小的时候，"空隙交叉处"的死细胞会凑巧得到 3 个邻居，
        于是被"生"出来，破坏了图案。
        实际测试：
            间距 3  → ❌ 12 格会变
            间距 4  → ✅ 完美静物
            间距 5  → ✅ 完美静物
        所以本程序用 **间距 4**（2 格方块 + 2 格空隙）。

    → 于是爱心被"像素化"成一个个 2×2 的小方块，整齐排列成心形。
      看起来是有点"点阵感"，但这正是稳定性的代价——
      实心爱心在生命游戏里一定会崩，只有这种骨架才能永存。

【初始状态怎么来的】
    和上一版一样的思路：爱心是"结果"，我们要找"原因"。
    用模拟退火反向搜索，找到一个 P 使  演化(P) == 爱心。
    这个 P 是密度 61.6% 的一团乱码，完全看不出爱心。

    搜索结果已经**直接写进本文件**（见 INITIAL_STATE），
    所以运行是**瞬间完成**的，不用等搜索。

【最终效果】
    第 0 步：一团乱码（61.6% 密度）
    第 1 步：方块爱心 ★
    第 2 步：一模一样
    第 3 步：一模一样
    ……
    永远不变
================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 一、生命游戏规则（和上一版相同）
# ============================================================
def life_step(grid):
    """
    Conway 生命游戏演化一步。

    规则（看每个格子周围的 8 个邻居）：
        活细胞 + 邻居 2 或 3 个  →  活
        活细胞 + 邻居 < 2 或 > 3  →  死
        死细胞 + 邻居正好 3 个    →  活

    实现：四周补一圈 0（棋盘外算死的），再把 8 个错位数组相加，
          得到的就是每个格子的邻居数。
    """
    p = np.pad(grid, 1)
    neighbors = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] +                p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    return ((neighbors == 3) | ((grid == 1) & (neighbors == 2))).astype(np.uint8)


# ============================================================
# 二、目标：稳定的方块爱心
# ============================================================
# 粗网格：B = 这里放一个 2x2 方块，. = 留空
# 每个 B 之间相距 PITCH=4 格（2 格方块 + 2 格空隙）

HEART_COARSE = """
.B...B.
BBB.BBB
BBBBBBB
.BBBBB.
..BBB..
...B...
"""

PITCH = 4          # 间距。实测 4 是能让爱心稳定的最小值
PADDING = 4        # 棋盘四周留白


def build_heart():
    """把粗网格展开成真实的细胞图案"""
    rows = [r.strip() for r in HEART_COARSE.strip('\n').split('\n') if r.strip()]
    ch, cw = len(rows), max(len(r) for r in rows)
    g = np.zeros((ch * PITCH, cw * PITCH), dtype=np.uint8)
    for r, line in enumerate(rows):
        for c in range(cw):
            if c < len(line) and line[c] == 'B':
                y, x = r * PITCH, c * PITCH
                g[y:y + 2, x:x + 2] = 1      # 放一个 2x2 方块
    return np.pad(g, PADDING)


# ============================================================
# 三、初始状态（模拟退火反向搜索的结果，已固化）
# ============================================================
# # = 活细胞，. = 死细胞
# 这是"演化一步就得到爱心"的前驱状态，搜索时用了约 2 分钟
INITIAL_STATE = """
#.####.######.#.#..#.#.###.######...
.########.####....#...#############.
###.##################..#..######...
####.##..##############.#....#####.#
#######....#####.######....#######.#
#########..#######.####.##...#####..
.########.###############.#########.
###.#....##.#..####...##..###..####.
####..#.##..##########..##..##.#####
.##.#......#...####.#..#..#....##.##
####..#.#.......#............#.#.###
###......#..#...#............#.####.
.#..#.......#........#..##.....####.
###.##..##...#..##..##...#..##.#####
###.....#.#..####.#.........#.##.###
###.####.#####..#......#...#########
#######.##..#...##...#....#.#.##..##
####.##.............##.#.#######.###
#######.#...##..#.....#..#..#.#####.
##..###.#.......#.....###########..#
###..#.###..##......##.#####.#.###..
.############...##..#..#######..###.
.##.##.####.....##.....######.####.#
####.#.####...##..########.#######..
#####.##.##.####.#.#####.#..##.###..
.#.####.#######...##########.#####.#
####..##.#####..#..###.######.#.##.#
####.#..#..#.##.#.######.#####.#.#..
#######.#####################.####.#
###..#############################.#
.#####...#...#..#......##..#######..
#.###.#.#...#....#.##........####.#.
"""


def parse(art):
    rows = [r.strip() for r in art.strip('\n').split('\n') if r.strip()]
    w = max(len(r) for r in rows)
    return np.array([[1 if c == '#' else 0 for c in r.ljust(w, '.')] for r in rows], dtype=np.uint8)


# ============================================================
# 四、主流程
# ============================================================
def main():
    print("=" * 62)
    print("生命游戏 · 爱心作图【稳定版】")
    print("=" * 62)

    target = build_heart()
    initial = parse(INITIAL_STATE)

    print(f"棋盘尺寸    : {target.shape[0]} x {target.shape[1]}")
    print(f"爱心活细胞  : {int(target.sum())} 个")
    print(f"初始密度    : {initial.mean():.1%}")
    print()

    if initial.shape != target.shape:
        print(f"[!] 尺寸不匹配：初始 {initial.shape} vs 目标 {target.shape}")
        return

    # ---------- 步骤 1：确认爱心自己是静物 ----------
    print("[1] 验证『爱心本身是静物』（演化一步后完全不变）")
    stable = np.array_equal(life_step(target), target)
    print(f"    结果: {'✅ 是静物' if stable else '❌ 会变'}")
    if not stable:
        return
    print()

    # ---------- 步骤 2：验证初始状态能演化出爱心 ----------
    print("[2] 验证『初始状态 → 爱心』")
    step1 = life_step(initial)
    ok = np.array_equal(step1, target)
    print(f"    演化 1 步 == 爱心 ?  {'✅ 是' if ok else '❌ 否'}")
    if not ok:
        return
    print()

    # ---------- 步骤 3：演示"永远不变" ----------
    print("[3] 连续演化，确认爱心永不变")
    g = initial.copy()
    checks = {}
    for n in range(1, 21):
        g = life_step(g)
        checks[n] = g.copy()
    same_as_1 = all(np.array_equal(checks[n], checks[1]) for n in range(1, 21))
    print(f"    第 1 步到第 20 步，全部与爱心相同 ?  {'✅ 是' if same_as_1 else '❌ 否'}")
    print()

    # ---------- 步骤 4：画演化过程图 ----------
    print("[4] 生成演化过程图 ...")
    show_steps = [0, 1, 2, 5, 10, 20]
    frames = [initial]
    g = initial.copy()
    for i in range(1, max(show_steps) + 1):
        g = life_step(g)
        frames.append(g.copy())

    cols = 3
    rows = (len(show_steps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, s in enumerate(show_steps):
        ax = axes[idx]
        ax.imshow(frames[s], cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])

        if s == 0:
            title, color = "第 0 步\n(乱码，61.6% 密度)", 'tab:blue'
        elif s == 1:
            title, color = "★ 第 1 步：爱心 ★", 'tab:red'
        else:
            title, color = f"第 {s} 步\n(与爱心完全相同)", 'tab:green'

        ax.set_title(title, color=color, fontsize=11,
                     fontweight='bold' if s == 1 else 'normal')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5 if s <= 1 else 1)

    for ax in axes[len(show_steps):]:
        ax.axis('off')

    fig.suptitle("生命游戏 · 稳定的爱心（出现后永不消散）",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("效果图_稳定版.png", dpi=150, bbox_inches='tight')
    print("    已保存 效果图_稳定版.png")
    print()

    # ---------- 步骤 5：打印爱心本身 ----------
    print("[5] 爱心图案（# = 活细胞）：")
    for row in target:
        print("    " + "".join("#" if c else "." for c in row))
    print()

    # ---------- 步骤 6：导出台式文件 ----------
    with open("爱心图案_稳定版.txt", "w", encoding="utf-8") as f:
        for row in target:
            f.write("".join("#" if c else "." for c in row) + "\n")
    print("    爱心图案已保存到 爱心图案_稳定版.txt")

    # 供网站在线验证用的格式（O=活, .=死）
    with open("爱心图案_稳定版_可在线验证.cells", "w", encoding="utf-8") as f:
        f.write("!Name: Stable Heart\n")
        for row in target:
            f.write("".join("O" if c else "." for c in row) + "\n")
    print("    在线验证用文件: 爱心图案_稳定版_可在线验证.cells")
    print()
    print("完成！")


if __name__ == "__main__":
    main()
