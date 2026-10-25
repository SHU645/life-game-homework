"""测试不同间距下，方块爱心是否稳定"""
import numpy as np


def life(g):
    p = np.pad(g, 1)
    n = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
         p[1:-1, :-2] + p[1:-1, 2:] +
         p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def expand(coarse_art, pitch):
    rows = [r.strip() for r in coarse_art.strip('\n').split('\n') if r.strip()]
    ch, cw = len(rows), max(len(r) for r in rows)
    g = np.zeros((ch * pitch, cw * pitch), dtype=np.uint8)
    for r, line in enumerate(rows):
        for c in range(cw):
            if c < len(line) and line[c] == 'B':
                g[r * pitch:r * pitch + 2, c * pitch:c * pitch + 2] = 1
    return g


HEART = """
.B...B.
BBB.BBB
BBBBBBB
.BBBBB.
..BBB..
...B...
"""

for pitch in [3, 4, 5, 6]:
    g = np.pad(expand(HEART, pitch), 8)
    nxt = life(g)
    diff = int(np.count_nonzero(nxt != g))
    mark = '✅ 静物' if diff == 0 else f'❌ {diff} 格会变'
    print(f"间距 {pitch}：尺寸 {g.shape}  活细胞 {int(g.sum())}  {mark}")
    if pitch <= 4:
        for row in g[6:-6]:
            s = ''.join('#' if c else '.' for c in row[6:-6])
            if s.strip('.'):
                print('   ' + s)
    print()
