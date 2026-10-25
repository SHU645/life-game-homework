"""测试：哪种爱心轮廓本身是生命游戏的'静物'（稳定不变）"""
import numpy as np


def life(g):
    p = np.pad(g, 1)
    n = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
         p[1:-1, :-2] + p[1:-1, 2:] +
         p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def parse(art):
    rows = [r for r in art.strip('\n').split('\n')]
    w = max(len(r) for r in rows)
    return np.array([[1 if c == 'O' else 0 for c in r.ljust(w, '.')] for r in rows], dtype=np.uint8)


def check(name, art):
    g = np.pad(parse(art), 6)          # 四周留白
    nxt = life(g)
    diff = int(np.count_nonzero(nxt != g))
    live = int(g.sum())
    print(f"\n【{name}】活细胞 {live}  不稳定格子 {diff}  {'✅ 静物！' if diff == 0 else '❌ 会变'}")
    for row in g:
        print('   ' + ''.join('#' if c else '.' for c in row))
    if diff:
        print("   演化一步后：")
        for row in nxt:
            print('   ' + ''.join('#' if c else '.' for c in row))
    return g, diff


# 方案 A：1 格粗的爱心轮廓
A = """
.OO...OO.
O..O.O..O
O...O...O
O.......O
.O.....O.
..O...O..
...O.O...
....O....
"""

# 方案 B：方块（已知静物，作对照）
B = """
OO
OO
"""

# 方案 C：蜂巢（已知静物，作对照）
C = """
.OO.
O..O
.OO.
"""

# 方案 D：心形用"块"拼（2x2 块彼此隔开）
D = """
OO...OO
OO...OO
........
........
"""

for name, art in [('A 单线轮廓', A), ('B 方块(对照)', B), ('C 蜂巢(对照)', C), ('D 方块爱心', D)]:
    check(name, art)
