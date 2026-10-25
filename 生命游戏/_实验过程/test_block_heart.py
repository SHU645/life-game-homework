"""用 2x2 方块拼爱心，方块间距 3 格（2 格方块 + 1 格空隙）"""
import numpy as np


def life(g):
    p = np.pad(g, 1)
    n = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
         p[1:-1, :-2] + p[1:-1, 2:] +
         p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def expand(coarse_art, pitch=3):
    """把粗网格（B=放方块）展开成真实细胞网格"""
    rows = [r.strip() for r in coarse_art.strip('\n').split('\n') if r.strip()]
    ch, cw = len(rows), max(len(r) for r in rows)
    out_h = ch * pitch
    out_w = cw * pitch
    g = np.zeros((out_h, out_w), dtype=np.uint8)
    for r, line in enumerate(rows):
        for c in range(cw):
            if c < len(line) and line[c] == 'B':
                y, x = r * pitch, c * pitch
                g[y:y + 2, x:x + 2] = 1
    return g


def test(name, art, pitch=3):
    g = np.pad(expand(art, pitch), 8)
    nxt = life(g)
    diff = int(np.count_nonzero(nxt != g))
    print(f"\n【{name}】尺寸 {g.shape}  活细胞 {int(g.sum())}  不稳定 {diff}  {'✅ 静物' if diff == 0 else '❌ 会变'}")
    for row in g:
        # 裁掉全空的行再打印，省空间
        print('   ' + ''.join('#' if c else '.' for c in row))
    return g, diff


# 粗网格爱心（7 宽 x 6 高）
H1 = """
.B...B.
BBB.BBB
BBBBBBB
.BBBBB.
..BBB..
...B...
"""

# 另一种：更圆润（7 宽 x 7 高）
H2 = """
.B...B.
BBBBBBB
BBBBBBB
BBBBBBB
.BBBBB.
..BBB..
...B...
"""

g1, d1 = test('方块爱心 v1', H1)
g2, d2 = test('方块爱心 v2', H2)

# 演化 3 步，确认真的不变
for nm, g, d in [('v1', g1, d1), ('v2', g2, d2)]:
    if d == 0:
        ok = all(np.array_equal(life(k), g) for k in [g, g, g])
        print(f"\n{nm} 连续演化 3 步后是否完全相同: {'✅ 是（真静物）' if np.array_equal(life(life(life(g))), g) else '❌ 否'}")
