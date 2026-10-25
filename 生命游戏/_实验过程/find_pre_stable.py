"""为'稳定的方块爱心'寻找前驱状态"""
import numpy as np
import time

PITCH = 4
HEART = """
.B...B.
BBB.BBB
BBBBBBB
.BBBBB.
..BBB..
...B...
"""


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


TARGET = np.pad(expand(HEART, PITCH), 4)
H, W = TARGET.shape
print(f"目标静物: {H}x{W}  活细胞 {int(TARGET.sum())}")
print(f"验证它自己是静物: {np.array_equal(life(TARGET), TARGET)}")


def search_predecessor(T, max_iter=600000, seed=0):
    rng = np.random.default_rng(seed)
    h, w = T.shape
    # 初始密度取目标密度
    p = (rng.random((h, w)) < max(T.mean(), 0.3)).astype(np.uint8)
    cur = int(np.count_nonzero(life(p) != T))
    best = cur
    temp = 1.5
    for _ in range(max_iter):
        if cur == 0:
            return p, 0, _
        mm = np.argwhere(life(p) != T)
        cy, cx = mm[rng.integers(len(mm))]
        y = (cy + rng.integers(-1, 2)) % h
        x = (cx + rng.integers(-1, 2)) % w
        p[y, x] ^= 1
        new = int(np.count_nonzero(life(p) != T))
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / temp):
            cur = new
            best = min(best, cur)
        else:
            p[y, x] ^= 1
        temp = max(temp * 0.999995, 0.05)
    return None, best, max_iter


print("\n=== 搜索前驱 ===")
t0 = time.time()
found = None
for seed in range(15):
    p, best, it = search_predecessor(TARGET, seed=seed)
    dt = time.time() - t0
    if p is not None:
        print(f"  seed={seed}  ✅ 找到！迭代 {it}  耗时 {dt:.1f}s")
        found = p
        break
    print(f"  seed={seed}  ❌ 未找到（最好剩 {best} 格不对）  {dt:.1f}s")

if found is not None:
    print(f"\n验证正向: {np.array_equal(life(found), TARGET)}")
    np.save(r'D:\代码\生命游戏\stable_heart_target.npy', TARGET)
    np.save(r'D:\代码\生命游戏\stable_heart_initial.npy', found)
    print("已保存两个 .npy 文件")
    print(f"\n初始状态（密度 {found.mean():.1%}）：")
    for row in found:
        print('  ' + ''.join('#' if c else '.' for c in row))
