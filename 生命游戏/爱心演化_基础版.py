# -*- coding: utf-8 -*-
"""
生命游戏 · 爱心作图作业
================================================================

【作业要求】
    1. 初始图片和终末图片不一样
    2. 终末图片是一个爱心
    3. 像素规则按照 Conway 的 Game of Life 执行

【核心难点：生命游戏是"不可逆"的】
    生命游戏只能正向推演：
        给定当前状态  ──规则──>  算出下一状态
    但反过来做不到：
        给定一个结果  ──X──>  反推出之前是什么

    而这道题要的恰恰是"倒着想"：
        我需要找一张"乱糟糟的初始图"，让它演化之后变成爱心。

    如果正面硬碰运气（随机生成初始图 → 演化 → 看是不是爱心），
    几乎不可能成功：18x19 的棋盘有 2^342 种状态，
    随机撞上"结果正好是爱心"的概率是天文数字分之一。

【解决思路：反向搜索】
    既然正向碰不到，那就反过来——从爱心出发，去找"它的前一步"。

    具体做法：
        1. 把爱心图案定义为"目标状态 T"
        2. 搜索一个状态 P，使得：  演化(P) == T
           这个 P 就叫 T 的"前驱状态"
        3. 找到的 P 是一团杂乱的像素，完全看不出爱心
           —— 这正是我们要的"初始状态"！

    这样得到的对应关系是：
        初始状态(乱码)  ──演化1步──>  爱心

【用什么算法找前驱：模拟退火】
    "演化(P) == T" 是一个约束满足问题，格子太多没法暴力枚举。
    我们用模拟退火（Simulated Annealing）：

        ① 随机生成一个 P
        ② 计算 演化(P)，和 T 逐格比较，数出"错了几个格子"（这就是"能量"）
        ③ 随机翻转 P 中的一个格子，重新计算错格数：
              - 错格变少  →  接受这次翻转
              - 错格变多  →  以一定概率接受（概率随"温度"降低而减小）
        ④ 重复，直到错格数 = 0，就找到了一个前驱

    第③步里"以一定概率接受变差的解"是关键，叫"退火"：
    它让搜索有机会跳出局部最优，不会卡死在"差一点点"的地方。

    另一个关键优化：**不要盲目随机翻格子**。
    棋盘有几百个格子，但真正"错"的只有几个，
    所以每次只在"错格周围 3x3 范围"里翻 ——
    效率提升了几十倍。（后面 search_predecessor 函数里有注释）

【为什么密度是个问题（这是我踩过的坑）】
    生命游戏有个统计规律：绝大多数图案演化几步后，
    密度会收敛到 2.87% 左右。
    而"爱心的前驱"是一张密度高达 60% 的图 ——
    这是非常罕见的状态。
    所以：
      · 反推 1 步 → 很容易（几秒）
      · 反推 2 步 → 极其困难（我试了 6 百万次迭代都没成功）
    这就是本方案只反推 1 步、而不做长链条的原因。

================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # 不弹窗，直接存文件（交作业更稳）
import matplotlib.pyplot as plt

# 让 matplotlib 能正常显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 一、爱心图案（这是我们的"目标状态" T）
# ============================================================
# 用 0/1 矩阵表示：1 = 活细胞，0 = 死细胞
# 一行一行地"画"出来，直观好改

HEART = np.array([
    [0, 1, 1, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
], dtype=np.uint8)

# 棋盘 = 爱心 + 四周留白
# 【为什么要留白】生命游戏演化时，图案会向外"扩散"。
# 如果爱心贴着棋盘边缘，边缘外被当作"死细胞"，
# 会让演化结果失真。留出足够空白才准确。
PADDING = 6
H, W = HEART.shape
BOARD = np.zeros((H + 2 * PADDING, W + 2 * PADDING), dtype=np.uint8)
BOARD[PADDING:PADDING + H, PADDING:PADDING + W] = HEART
TARGET = BOARD          # 目标状态


# ============================================================
# 二、生命游戏的核心规则
# ============================================================
def life_step(grid):
    """
    把整个棋盘向前演化一步。

    Conway 生命游戏的规则（对每个格子，看它周围 8 个邻居）：

        1. 活细胞 + 邻居 2 或 3 个   →  继续活着
        2. 活细胞 + 邻居 < 2 个      →  孤独而死
        3. 活细胞 + 邻居 > 3 个      →  拥挤而死
        4. 死细胞 + 邻居正好 3 个    →  繁殖，变活

    实现技巧：
        先用 np.pad 在四周补一圈 0（表示棋盘外是死细胞），
        然后把这 8 个"错位后的数组"直接相加，
        得到的就是每个格子的"邻居数"。
        这比写双重循环快几十倍，代码也更短。
    """
    p = np.pad(grid, 1)                       # 四周补一圈 0
    neighbors = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +      # 上排 3 个
        p[1:-1, :-2] +               p[1:-1, 2:] +      # 左右 2 个
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]            # 下排 3 个
    )
    alive = (neighbors == 3) | ((grid == 1) & (neighbors == 2))
    return alive.astype(np.uint8)


def life_n(grid, n):
    """连续演化 n 步"""
    for _ in range(n):
        grid = life_step(grid)
    return grid


# ============================================================
# 三、核心：反向搜索，找一个"前驱状态"
# ============================================================
def search_predecessor(target, max_iter=400_000, seed=0):
    """
    用模拟退火搜索满足  life(P) == target  的状态 P。

    参数：
        target   —— 目标图案（这里就是爱心）
        max_iter —— 最大迭代次数
        seed     —— 随机种子（换个种子相当于换一次运气）

    返回：
        找到的前驱 P；找不到则返回 None

    ------------------------------------------------
    实现要点（这几条是效率的关键）：

    ① 能量函数 = 错格数
       用  np.count_nonzero(life(P) != target)  衡量"离目标还差多远"。
       错格数 = 0 就是找到了。

    ② 只翻"错格附近"的格子
       生命游戏里，翻转格子 (y,x) 只会影响 y-1..y+1、x-1..x+1
       这一小块区域的下一状态。
       所以离错格很远的地方，翻了也是白翻。
       我们把随机翻转的范围限制在"某个错格的 3x3 邻域"，
       收敛速度比盲翻快了几十倍。

    ③ 温度慢慢降（退火）
       温度高时，愿意接受"变差"的翻转，便于大范围探索；
       温度低时，几乎只接受"变好"的翻转，便于精细收敛。
       这里用等比降温：temp *= 0.99999，并设下限 0.02，
       保留一点点"随机性"，避免彻底卡死。
    ------------------------------------------------
    """
    rng = np.random.default_rng(seed)
    h, w = target.shape

    # ① 随机初始化（起始密度取 0.35 左右，是反复试出来的好起点）
    p = (rng.random((h, w)) < 0.35).astype(np.uint8)
    error = int(np.count_nonzero(life_step(p) != target))

    temp = 1.2                     # 初始温度

    for _ in range(max_iter):
        if error == 0:             # 找到了
            return p

        # ② 找出当前所有错格，随机挑一个
        wrong = np.argwhere(life_step(p) != target)
        cy, cx = wrong[rng.integers(len(wrong))]

        # ③ 在它的 3x3 邻域内随机翻一个格子
        y = (cy + rng.integers(-1, 2)) % h
        x = (cx + rng.integers(-1, 2)) % w
        p[y, x] ^= 1               # 翻转（0变1 或 1变0）

        new_error = int(np.count_nonzero(life_step(p) != target))
        delta = new_error - error

        # ④ 退火接受准则
        if delta <= 0 or rng.random() < np.exp(-delta / temp):
            error = new_error      # 接受
        else:
            p[y, x] ^= 1           # 反悔，翻回来

        temp = max(temp * 0.99999, 0.02)

    return None                    # 迭代用尽仍未找到


def find_initial_state(target, max_tries=20):
    """换不同随机种子多试几次，提高成功率"""
    for seed in range(max_tries):
        res = search_predecessor(target, seed=seed)
        if res is not None:
            print(f"     第 {seed + 1} 次尝试成功")
            return res
    return None


# ============================================================
# 四、主流程
# ============================================================
def main():
    print("=" * 60)
    print("生命游戏 · 爱心作图")
    print("=" * 60)
    print(f"棋盘尺寸 : {TARGET.shape[0]} x {TARGET.shape[1]}")
    print(f"爱心活细胞: {int(HEART.sum())} 个")
    print()

    # ---------- 步骤 1：反向搜索初始状态 ----------
    print("[1] 正在反向搜索『演化一步就能得到爱心』的初始状态 ...")
    import time
    t0 = time.time()
    initial = find_initial_state(TARGET)
    if initial is None:
        print("    搜索失败，请重试或调整参数")
        return
    print(f"    找到！耗时 {time.time() - t0:.1f} 秒")
    print(f"    初始状态密度: {initial.mean():.1%}（爱心是 {TARGET.mean():.1%}）")
    print()

    # ---------- 步骤 2：正向验证 ----------
    # 这一步很重要：不能只相信搜索结果，必须真的演化一遍确认
    print("[2] 正向验证：从初始状态开始演化，看是否真的出现爱心")
    ok = np.array_equal(life_step(initial), TARGET)
    print(f"    演化 1 步 == 爱心 ?  {'✅ 是' if ok else '❌ 否'}")
    if not ok:
        return
    print()

    # ---------- 步骤 3：生成演化序列，做成图 ----------
    # 展示：初始状态 →（演化）→ 爱心 →（继续演化）→ …
    # 说明一下：爱心只在第 1 步出现。
    # 之后继续演化是为了展示"这确实是真实的生命游戏"，
    # 而不是一张静止的贴图。
    print("[3] 生成演化过程图 ...")
    STEPS_AFTER = 5
    frames = [initial]
    g = initial.copy()
    for _ in range(1 + STEPS_AFTER):
        g = life_step(g)
        frames.append(g.copy())

    n = len(frames)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes).flatten()

    for i, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])

        if i == 0:
            title, color = "初始状态\n(第 0 步)", 'tab:blue'
        elif i == 1:
            title, color = "★ 第 1 步：爱心 ★", 'tab:red'
        else:
            title, color = f"第 {i} 步", 'gray'

        ax.set_title(title, color=color, fontsize=11,
                     fontweight='bold' if i == 1 else 'normal')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2 if i == 1 else 0.5)

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle("Conway 生命游戏 · 从随机状态演化出爱心",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("效果图_基础版.png", dpi=150, bbox_inches='tight')
    print("    已保存 效果图_基础版.png")
    print()

    # ---------- 步骤 4：把初始状态打印出来（方便贴进作业文档） ----------
    print("[4] 初始状态（# = 活细胞，. = 死细胞）：")
    print()
    for row in initial:
        print("    " + "".join("#" if c else "." for c in row))
    print()

    # ---------- 步骤 5：保存初始状态为文本，方便复现 ----------
    with open("初始状态_基础版.txt", "w", encoding="utf-8") as f:
        for row in initial:
            f.write("".join("#" if c else "." for c in row) + "\n")
    print("    初始状态已保存到 初始状态_基础版.txt")
    np.save("初始状态_基础版.npy", initial)
    print("    初始状态已保存到 初始状态_基础版.npy")
    print()
    print("完成！")


if __name__ == "__main__":
    main()
