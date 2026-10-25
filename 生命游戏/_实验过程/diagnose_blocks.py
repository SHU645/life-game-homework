"""定位方块爱心里到底是哪些格子不稳定、为什么"""
import numpy as np


def life(g):
    p = np.pad(g, 1)
    n = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
         p[1:-1, :-2] + p[1:-1, 2:] +
         p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def neigh(g):
    p = np.pad(g, 1)
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])


def expand(coarse_art, pitch=3):
    rows = [r.strip() for r in coarse_art.strip('\n').split('\n') if r.strip()]
    ch, cw = len(rows), max(len(r) for r in rows)
    g = np.zeros((ch * pitch, cw * pitch), dtype=np.uint8)
    for r, line in enumerate(rows):
        for c in range(cw):
            if c < len(line) and line[c] == 'B':
                g[r * pitch:r * pitch + 2, c * pitch:c * pitch + 2] = 1
    return g


H = """
.B...B.
BBB.BBB
BBBBBBB
.BBBBB.
..BBB..
...B...
"""

g = np.pad(expand(H), 8)
n = neigh(g)
nxt = life(g)
bad = np.argwhere(nxt != g)
print(f"不稳定格子共 {len(bad)} 个：\n")
print("   位置(行,列)  当前状态  邻居数  下一步")
for (y, x) in bad:
    state = '活' if g[y, x] else '死'
    print(f"   ({y:>2},{x:>2})      {state}        {n[y,x]}      {'活' if nxt[y,x] else '死'}")

print("\n=== 规则 ===")
print("  活细胞需要 2 或 3 个邻居；死细胞需要 ≠3 个邻居")

# 打印受影响区域的放大图
ys = [y for y, x in bad]
xs = [x for y, x in bad]
y0, y1 = max(0, min(ys) - 2), min(g.shape[0], max(ys) + 3)
x0, x1 = max(0, min(xs) - 2), min(g.shape[1], max(xs) + 3)
print(f"\n=== 问题区域（行 {y0}-{y1}, 列 {x0}-{x1}）===")
print("    当前状态:")
for y in range(y0, y1):
    line = ''
    for x in range(x0, x1):
        ch = '#' if g[y, x] else '.'
        if (y, x) in set(map(tuple, bad)):
            ch = 'X'          # 标出问题格
        line += ch
    print(f"   {y:>2} {line}")
