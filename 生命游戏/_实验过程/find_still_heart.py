"""搜索一个'看起来像爱心'的静物（still life）"""
import numpy as np
import time


def life(g):
    p = np.pad(g, 1)
    n = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
         p[1:-1, :-2] + p[1:-1, 2:] +
         p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def parse(art):
    rows = art.strip('\n').split('\n')
    w = max(len(r) for r in rows)
    return np.array([[1 if c == 'O' else 0 for c in r.ljust(w, '.')] for r in rows], dtype=np.uint8)


def show(g, title):
    print(f"\n{title}")
    for row in g:
        print('   ' + ''.join('#' if c else '.' for c in row))


HEART = """
.OO...OO.
O..O.O..O
O...O...O
O.......O
.O.....O.
..O...O..
...O.O...
....O....
"""

H = np.pad(parse(HEART), 6)
h, w = H.shape
print(f"目标爱心: {h}x{w}, 活细胞 {H.sum()}")


def anneal(target, shape_weight=8.0, iters=400000, seed=0):
    """最小化: 不稳定格子数 * 大权重 + 与原心的差异"""
    rng = np.random.default_rng(seed)
    p = target.copy()
    # 从目标出发，稍微打乱
    noise = (rng.random((h, w)) < 0.08).astype(np.uint8)
    p = p ^ noise

    def energy(g):
        inst = int(np.count_nonzero(life(g) != g))
        ham = int(np.count_nonzero(g != target))
        return inst * 100 + ham * shape_weight, inst, ham

    cur, cur_inst, cur_ham = energy(p)
    best_p, best_e = p.copy(), cur
    temp = 3.0

    for _ in range(iters):
        if cur_inst == 0 and cur_ham < 20:
            return p, 0, cur_ham, cur
        y, x = rng.integers(0, h), rng.integers(0, w)
        p[y, x] ^= 1
        e, inst, ham = energy(p)
        d = e - cur
        if d <= 0 or rng.random() < np.exp(-d / temp):
            cur, cur_inst, cur_ham = e, inst, ham
            if e < best_e:
                best_e, best_p = e, p.copy()
        else:
            p[y, x] ^= 1
        temp = max(temp * 0.99999, 0.05)
    return best_p, cur_inst, cur_ham, best_e


print("\n=== 开始搜索 ===")
t0 = time.time()
results = []
for sw in [12.0, 8.0, 5.0, 3.0, 1.5]:
    for seed in range(3):
        p, inst, ham, e = anneal(H, shape_weight=sw)
        if inst == 0:
            results.append((ham, sw, seed, p.copy()))
            print(f"  shape_w={sw:<5} seed={seed}  ✅ 找到静物！与原心差异 {ham} 格")
            break
    else:
        print(f"  shape_w={sw:<5} ❌ 没找到静物")
    if len(results) >= 3:
        break

print(f"\n耗时 {time.time()-t0:.1f}s")

if results:
    results.sort(key=lambda r: r[0])
    ham, sw, seed, best = results[0]
    show(best, f"最像爱心的静物（shape_w={sw}, 差异 {ham} 格）：")
    # 验证
    print(f"\n验证：演化一步后是否完全不变？ {'✅ 是' if np.array_equal(life(best), best) else '❌ 否'}")
else:
    print("\n没找到。需要换思路。")
