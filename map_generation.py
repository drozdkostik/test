# coding: utf-8
"""
Карта (X=200 км, Y=50 км) зі стартом БпЛА (20 км, 50 км), ціллю (45 км, 190 км),
20 окремих ОП і 12 РОП (кожен 2–4 ОП у радіусі ≤200 м). Дотримано:
- верхньорівневі центри (20 ОП + 12 центрів РОП) мають кроки 7–10 км від принаймні одного сусіда
  і мінімум 7 км до решти;
- перша точка ≤10 км від старту; остання точка ≤10 км від цілі;
- старт — синій трикутник; ціль — зірка;
- масштабування колесом та панорамування правою кнопкою миші.

Оси: X ∈ [0, 200_000] м (200 км), Y ∈ [0, 50_000] м (50 км).
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ------------------------- Параметри сцени -------------------------
MAP_WIDTH_M = 200_000      # 200 км уздовж X
MAP_HEIGHT_M = 50_000      # 50 км уздовж Y

# Точки пуску та цілі (км -> м)
START_KM = (20.0, 50.0)
TARGET_KM = (190.0, 45.0)
START_M = (START_KM[0] * 1000.0, START_KM[1] * 1000.0)
TARGET_M = (TARGET_KM[0] * 1000.0, TARGET_KM[1] * 1000.0)

# Кількості
N_STANDALONE_OP = 20
N_ROP = 12
ROP_MIN_OP = 2
ROP_MAX_OP = 4

# Розміри ОП (метри)
OP_WIDTH_MIN, OP_WIDTH_MAX = 10.0, 20.0
OP_LENGTH_MIN, OP_LENGTH_MAX = 12.0, 25.0

# ОП у РОП (радіус області)
ROP_RADIUS_MAX = 200.0  # м

# Відстані верхнього рівня (метри)
MIN_GAP_M = 7_000.0        # мінімальна відстань до будь-якого іншого центру
MAX_LINK_M = 10_000.0      # повинен існувати принаймні один сусід <= 10 км
FIRST_MAX_FROM_START = 10_000.0  # перша точка ≤ 10 км від старту
LAST_MAX_FROM_TARGET = 10_000.0  # остання точка ≤ 10 км від цілі

# Внутрішні налаштування
RNG_SEED = 42
MAX_ATTEMPTS_TOP = 300_000
MARGIN_M = 1_000.0  # запас від країв для верхньорівневих точок

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ------------------------- Структури даних -------------------------
@dataclass
class OP:
    center: Tuple[float, float]  # (x, y) м
    width: float                 # коротка сторона, м
    length: float                # довга сторона, м
    angle_deg: float             # орієнтація, градуси [0, 180)
    parent_rop: int | None = None  # індекс РОП або None

@dataclass
class ROP:
    center: Tuple[float, float]
    ops: List[OP]

# ------------------------- Допоміжні функції -------------------------
def within_map(pt: Tuple[float, float], margin=MARGIN_M) -> bool:
    x, y = pt
    return (margin <= x <= MAP_WIDTH_M - margin) and (margin <= y <= MAP_HEIGHT_M - margin)

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def random_bearing() -> float:
    return random.uniform(0.0, 2 * math.pi)

def step_from(base: Tuple[float, float], dist: float, bearing_rad: float) -> Tuple[float, float]:
    return (base[0] + dist * math.cos(bearing_rad), base[1] + dist * math.sin(bearing_rad))

def sample_op_size_and_angle() -> Tuple[float, float, float]:
    width = random.uniform(OP_WIDTH_MIN, OP_WIDTH_MAX)
    length = random.uniform(OP_LENGTH_MIN, OP_LENGTH_MAX)
    angle = random.uniform(0.0, 180.0)
    return width, length, angle

def sample_in_circle(center: Tuple[float, float], r_max: float) -> Tuple[float, float]:
    r = math.sqrt(random.random()) * r_max  # рівномірно по площі
    ang = random_bearing()
    return (center[0] + r * math.cos(ang), center[1] + r * math.sin(ang))

# ------------------------- Ланцюг START → TARGET -------------------------
def build_chain_from_start_to_target(step_min=MIN_GAP_M, step_max=MAX_LINK_M, jitter_deg=20.0) -> List[Tuple[float, float]]:
    """
    1) Перша точка — в колі R=10 км від START_M.
    2) Рухаємось у бік цілі кроками 7–10 км з випадковим відхиленням напряму ±jitter_deg.
    3) Коли відстань до TARGET_M ≤ 10 км — додаємо фінальну точку в колі R=10 км від TARGET_M,
       що також виконує мінімальні відстані до решти та має сусіда ≤10 км.
    """
    pts: List[Tuple[float, float]] = []

    # 1) Перша точка
    for _ in range(MAX_ATTEMPTS_TOP):
        cand = sample_in_circle(START_M, FIRST_MAX_FROM_START)
        if within_map(cand):
            pts.append(cand)
            break
    if not pts:
        raise RuntimeError("Не вдалося розмістити першу точку біля старту.")

    def azimuth(a, b):
        return math.atan2(b[1] - a[1], b[0] - a[0])

    # 2) Кроки до зони цілі
    while distance(pts[-1], TARGET_M) > LAST_MAX_FROM_TARGET:
        last = pts[-1]
        base_bearing = azimuth(last, TARGET_M)
        d = random.uniform(step_min, step_max)
        b = base_bearing + math.radians(random.uniform(-jitter_deg, jitter_deg))
        cand = step_from(last, d, b)

        # якщо вийшли за межі — кілька спроб скорегувати
        ok = False
        for _ in range(30):
            if within_map(cand) and all(distance(cand, p) >= MIN_GAP_M for p in pts):
                ok = True
                break
            b = base_bearing + math.radians(random.uniform(-jitter_deg, jitter_deg))
            d = random.uniform(step_min, step_max)
            cand = step_from(last, d, b)
        if not ok:
            # запасний варіант — невеликий крок до центру карти
            center_pt = (MAP_WIDTH_M/2, MAP_HEIGHT_M/2)
            b = azimuth(last, center_pt)
            cand = step_from(last, step_min, b)
            if not within_map(cand) or any(distance(cand, p) < MIN_GAP_M for p in pts):
                break  # вийти, щоб не зациклитись

        pts.append(cand)

    # 3) Остання точка в районі цілі
    placed_last = False
    for _ in range(MAX_ATTEMPTS_TOP):
        cand = sample_in_circle(TARGET_M, LAST_MAX_FROM_TARGET)
        if within_map(cand) and all(distance(cand, p) >= MIN_GAP_M for p in pts) and any(distance(cand, p) <= MAX_LINK_M for p in pts):
            pts.append(cand)
            placed_last = True
            break

    if not placed_last:
        # Фолбек: беримо найближчу до цілі точку і ставимо останню на 9.5 км у бік цілі
        nearest = min(pts, key=lambda p: distance(p, TARGET_M))
        base_bearing = azimuth(nearest, TARGET_M)
        for _ in range(3000):
            b = base_bearing + math.radians(random.uniform(-10, 10))
            d = min(MAX_LINK_M * 0.95, max(MIN_GAP_M, distance(nearest, TARGET_M) * 0.9))
            cand = step_from(nearest, d, b)
            if within_map(cand) and distance(cand, TARGET_M) <= LAST_MAX_FROM_TARGET and all(distance(cand, p) >= MIN_GAP_M for p in pts):
                pts.append(cand)
                placed_last = True
                break
        if not placed_last:
            raise RuntimeError("Не вдалося розмістити останню точку біля цілі.")

    return pts

# ------------------------- Генерація сцени -------------------------
def generate_scene():
    need_top = N_STANDALONE_OP + N_ROP  # 20 + 12 = 32
    pts = build_chain_from_start_to_target()

    # Якщо точок у ланцюзі замало — дозаповнити, дотримуючись правил
    attempts = 0
    while len(pts) < need_top and attempts < MAX_ATTEMPTS_TOP:
        attempts += 1
        base = random.choice(pts)
        d = random.uniform(MIN_GAP_M, MAX_LINK_M)
        b = random_bearing()
        cand = step_from(base, d, b)
        if within_map(cand) and all(distance(cand, p) >= MIN_GAP_M for p in pts) and any(distance(cand, p) <= MAX_LINK_M for p in pts):
            pts.append(cand)

    if len(pts) < need_top:
        raise RuntimeError("Не вдалося добрати потрібну кількість верхньорівневих точок.")

    # Розподіл: перші 20 — окремі ОП; наступні 12 — центри РОП
    standalone_centers = pts[:N_STANDALONE_OP]
    rop_centers = pts[N_STANDALONE_OP:N_STANDALONE_OP + N_ROP]

    standalone_ops: List[OP] = []
    for c in standalone_centers:
        w, l, a = sample_op_size_and_angle()
        standalone_ops.append(OP(center=c, width=w, length=l, angle_deg=a, parent_rop=None))

    rops: List[ROP] = []
    for i, rc in enumerate(rop_centers):
        k = random.randint(ROP_MIN_OP, ROP_MAX_OP)
        ops: List[OP] = []
        for _ in range(k):
            op_center = sample_in_circle(rc, ROP_RADIUS_MAX)
            # Кліп до меж карти
            x = min(max(op_center[0], 0.0), MAP_WIDTH_M)
            y = min(max(op_center[1], 0.0), MAP_HEIGHT_M)
            op_center = (x, y)
            w, l, a = sample_op_size_and_angle()
            ops.append(OP(center=op_center, width=w, length=l, angle_deg=a, parent_rop=i))
        rops.append(ROP(center=rc, ops=ops))

    return standalone_ops, rops

# ------------------------- Візуалізація з масштабуванням -------------------------
def rect_corners(center: Tuple[float, float], w: float, l: float, angle_deg: float) -> np.ndarray:
    cx, cy = center
    a = math.radians(angle_deg)
    dx = l / 2.0
    dy = w / 2.0
    local = np.array([[+dx, +dy], [-dx, +dy], [-dx, -dy], [+dx, -dy], [+dx, +dy]])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    world = (local @ R.T) + np.array([[cx, cy]])
    return world

def plot_scene(standalone_ops: List[OP], rops: List[ROP]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.2))  # співвідношення ~200:50 = 4:1

    # Рамка карти
    ax.plot([0, MAP_WIDTH_M, MAP_WIDTH_M, 0, 0], [0, 0, MAP_HEIGHT_M, MAP_HEIGHT_M, 0], linewidth=1)

    # Старт і ціль
    ax.plot(START_M[0], START_M[1], marker='^', markersize=10, color='blue')  # синій трикутник
    ax.text(START_M[0], START_M[1], ' Старт', va='bottom', ha='left', color='blue')
    ax.plot(TARGET_M[0], TARGET_M[1], marker='*', markersize=12)
    ax.text(TARGET_M[0], TARGET_M[1], ' Ціль', va='bottom', ha='left')

    # Окремі ОП
    for op in standalone_ops:
        pts = rect_corners(op.center, op.width, op.length, op.angle_deg)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1)
        ax.plot(op.center[0], op.center[1], marker='o', markersize=3)

    # РОП та їх ОП
    for rop in rops:
        ax.plot(rop.center[0], rop.center[1], marker='x', markersize=6)
        for op in rop.ops:
            pts = rect_corners(op.center, op.width, op.length, op.angle_deg)
            ax.plot(pts[:, 0], pts[:, 1], linewidth=1)
            ax.plot(op.center[0], op.center[1], marker='o', markersize=2)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Карта X=200 км, Y=50 км: старт, ціль, 20 ОП та 12 РОП (метри)')
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    plt.tight_layout()

    # --- Інтерактивний зум колесом миші ---
    def on_scroll(event):
        if event.inaxes != ax:
            return
        scale_factor = 1.2 if event.button == 'up' else (1/1.2)
        cur_xmin, cur_xmax = ax.get_xlim()
        cur_ymin, cur_ymax = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w = (cur_xmax - cur_xmin) * scale_factor
        new_h = (cur_ymax - cur_ymin) * scale_factor
        relx = (xdata - cur_xmin) / (cur_xmax - cur_xmin)
        rely = (ydata - cur_ymin) / (cur_ymax - cur_ymin)
        new_xmin = xdata - relx * new_w
        new_xmax = new_xmin + new_w
        new_ymin = ydata - rely * new_h
        new_ymax = new_ymin + new_h
        ax.set_xlim(new_xmin, new_xmax)
        ax.set_ylim(new_ymin, new_ymax)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # --- Панорамування правою кнопкою миші ---
    state = {'press': None}
    def on_button_press(event):
        if event.inaxes != ax:
            return
        if event.button == 3:
            state['press'] = (event.xdata, event.ydata, ax.get_xlim(), ax.get_ylim())

    def on_motion(event):
        if state['press'] is None or event.inaxes != ax:
            return
        x0, y0, (xmin, xmax), (ymin, ymax) = state['press']
        dx = event.xdata - x0
        dy = event.ydata - y0
        ax.set_xlim(xmin - dx, xmax - dx)
        ax.set_ylim(ymin - dy, ymax - dy)
        fig.canvas.draw_idle()

    def on_button_release(event):
        if event.button == 3:
            state['press'] = None

    fig.canvas.mpl_connect('button_press_event', on_button_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_button_release)

    plt.show()

# ------------------------- Перевірка обмежень -------------------------
def validate_top_level(standalone_ops: List[OP], rops: List[ROP]):
    centers = [op.center for op in standalone_ops] + [r.center for r in rops]
    n = len(centers)
    dmat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(centers[i], centers[j])
            dmat[i, j] = dmat[j, i] = d
    mins = np.nanmin(dmat + np.eye(n) * 1e18, axis=1)
    print("Мінімальна відстань до сусіда (м):", np.round(mins, 2))
    print("Мінімум:", float(np.nanmin(mins)), "| Середнє:", float(np.nanmean(mins)), "| Максимум:", float(np.nanmax(mins)))
    ok_min = np.all(mins >= MIN_GAP_M - 1e-6)
    print("Виконано умову мін. відстані ≥ 7 км для всіх?", bool(ok_min))
    print("Відстань першої точки до старту (м):", distance(centers[0], START_M))
    print("Відстань останньої точки до цілі (м):", distance(centers[-1], TARGET_M))

# ------------------------- Точка входу -------------------------
if __name__ == "__main__":
    standalone_ops, rops = generate_scene()

    print(f"Окремих ОП: {len(standalone_ops)}")
    print(f"РОП: {len(rops)} (усього ОП у РОП: {sum(len(r.ops) for r in rops)})")

    validate_top_level(standalone_ops, rops)

    try:
        plot_scene(standalone_ops, rops)
    except Exception as e:
        print("Помилка побудови графіка (можливо, відсутній matplotlib):", e)
