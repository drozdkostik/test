# Імпорт необхідних бібліотек
import csv  # бібліотека для роботи з CSV файлами
import math  # математичні функції
import os  # для роботи з файловою системою
from typing import Tuple, Sequence, List, Optional, Dict  # типи даних

import numpy as np  # бібліотека для роботи з масивами
from matplotlib import pyplot as plt  # бібліотека для візуалізації
from matplotlib.axes import Axes  # осі для графіків
from matplotlib.collections import LineCollection  # колекція ліній
from matplotlib.patches import Circle  # клас для малювання кіл
# Імпорт власних модулів
from map_generation import EllipseParams, Point, BattleLine, MapConfig, CircleSpec, CorridorConfig, \
    compute_ellipse_params, zone_boundaries, generate_ops_in_zones, generate_rop_clusters, generate_op_circles, \
    generate_battle_line, rotate_local_to_global, rotate_global_to_local


def calculate_line_length(battle_line: BattleLine) -> float:
    """Обчислює загальну довжину лінії бойового зіткнення."""
    total_length = 0.0  # початкова довжина
    for i in range(len(battle_line.x) - 1):  # для всіх послідовних пар точок
        dx = battle_line.x[i + 1] - battle_line.x[i]  # різниця x-координат
        dy = battle_line.y[i + 1] - battle_line.y[i]  # різниця y-координат
        total_length += math.hypot(dx, dy)  # додавання довжини сегмента
    return float(total_length)


def save_battle_line_to_csv(battle_line: BattleLine, filename: str) -> None:
    """Зберігає координати лінії бойового зіткнення у CSV файл."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # створення директорії
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["X", "Y", "Width"])  # заголовки
        for x, y, w in zip(battle_line.x, battle_line.y, battle_line.widths):
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{w:.6f}"])  # запис рядків


def save_battle_line_detailed(battle_line: BattleLine, cfg: MapConfig, filename: str) -> None:
    """Зберігає детальну інформацію про лінію бойового зіткнення."""
    import datetime

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Запис метаданих
        writer.writerow(["# Координати лінії бойового зіткнення"])
        writer.writerow([f"# Дата створення: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([f"# Розмір карти: {cfg.width_km}x{cfg.height_km} км"])
        writer.writerow([f"# Кількість точок: {len(battle_line.x)}"])
        writer.writerow([f"# Довжина лінії: {calculate_line_length(battle_line):.2f} км"])
        writer.writerow([""])
        # Запис даних точок
        writer.writerow(["Номер_точки", "X_координата_км", "Y_координата_км", "Товщина_лінії_км", "Азимут_градуси"])

        for i, (x, y, w) in enumerate(zip(battle_line.x, battle_line.y, battle_line.widths), 1):
            azimuth = 0.0
            if i < len(battle_line.x):
                dx = battle_line.x[i] - battle_line.x[i - 1] if i > 0 else 0.0
                dy = battle_line.y[i] - battle_line.y[i - 1] if i > 0 else 0.0
                if dx != 0.0 or dy != 0.0:
                    azimuth = math.degrees(math.atan2(dy, dx))
                    if azimuth < 0:
                        azimuth += 360.0
            writer.writerow([i, f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{azimuth:.2f}"])


def save_ops_csv(ops: List[Point], filename: str) -> None:
    """Зберігає список ОП у CSV."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y"])  # заголовки
        for x, y in ops:
            w.writerow([f"{x:.6f}", f"{y:.6f}"])  # координати точок


def save_circles_csv(circles: List[CircleSpec], filename: str, header: Tuple[str, str, str] = ("X", "Y", "R")) -> None:
    """Зберігає параметри кіл у CSV."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for (x, y), r in circles:
            w.writerow([f"{x:.6f}", f"{y:.6f}", f"{r:.6f}"])  # центр та радіус


def _draw_grid(ax: Axes, cfg: MapConfig) -> None:
    """Малює рамку та сітку на графіку."""
    ax.set_xlim(0.0, cfg.width_km)
    ax.set_ylim(0.0, cfg.height_km)
    ax.set_aspect("equal", adjustable="box")

    xs = np.arange(0.0, cfg.width_km + cfg.cell_size, cfg.cell_size)  # точки сітки по x
    ys = np.arange(0.0, cfg.height_km + cfg.cell_size, cfg.cell_size)  # точки сітки по y
    for x in xs:
        ax.axvline(x, lw=0.5, alpha=0.2)  # вертикальні лінії
    for y in ys:
        ax.axhline(y, lw=0.5, alpha=0.2)  # горизонтальні лінії

    ax.set_xlabel("X, км")
    ax.set_ylabel("Y, км")


def _draw_ellipse(ax: Axes, p: EllipseParams, **kwargs) -> None:
    """Малює контур еліпса."""
    t = np.linspace(0.0, 2 * math.pi, 720)  # параметричні точки
    x_local = p.a * np.cos(t)  # локальні x-координати
    y_local = p.b * np.sin(t)  # локальні y-координати
    xg, yg = rotate_local_to_global(x_local, y_local, p)  # перетворення в глобальні
    ax.plot(xg, yg, **({"lw": 1.2, "alpha": 0.9} | kwargs))  # малювання контуру


def _draw_points(ax: Axes, pts: List[Point], label: str, size: float = 10.0, alpha: float = 0.85) -> None:
    """Малює точки на графіку."""
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=size, alpha=alpha, label=label)


def _draw_circles(ax: Axes, circles: List[CircleSpec], lw: float = 0.8, alpha: float = 0.8,
                  label: Optional[str] = None) -> None:
    """Малює кола на графіку."""
    for (cx, cy), r in circles:
        circ = Circle((cx, cy), r, fill=False, lw=lw, alpha=alpha)
        ax.add_patch(circ)
    if label:
        ax.plot([], [], lw=lw, alpha=alpha, label=label)  # точка для легенди


def _draw_battle_line(ax: Axes, bl: BattleLine) -> None:
    """Малює лінію бою."""
    points = np.column_stack([bl.x, bl.y])  # масив точок
    segments = np.stack([points[:-1], points[1:]], axis=1)  # сегменти
    lw_scale = 10.0
    lws = lw_scale * (bl.widths[:-1] / np.max(bl.widths))  # масштабування товщин
    lc = LineCollection(segments, linewidths=lws, alpha=0.9)  # колекція ліній
    ax.add_collection(lc)


def build_map(cfg: MapConfig, cor: CorridorConfig) -> Dict[str, object]:
    """Будує всі елементи карти."""
    rng = np.random.default_rng(cor.random_seed)

    p = compute_ellipse_params(cfg.start, cfg.goal, cor.b)  # параметри еліпса

    z_bounds, _, _ = zone_boundaries(p.a, cor.max_zone_width)  # межі зон
    ops = generate_ops_in_zones(p, z_bounds, rng, cor.points_per_zone)  # базові ОП

    rop_ops, rop_circles = generate_rop_clusters(  # РОП-кластери
        p=p,
        z_bounds=z_bounds,
        rng=rng,
        step=cor.rop_zone_step,
        radius_range=cor.rop_radius_range,
        points_range=cor.rop_points_range,
        existing_ops=ops,
    )
    ops_all = ops + rop_ops

    op_circles = generate_op_circles(ops_all, rng)  # кола навколо ОП

    battle_line = generate_battle_line(cfg, seed=cor.random_seed or 42)  # лінія бою

    return {
        "ellipse": p,
        "zones": z_bounds,
        "ops_base": ops,
        "ops_rop": rop_ops,
        "ops_all": ops_all,
        "rop_circles": rop_circles,
        "op_circles": op_circles,
        "battle_line": battle_line,
    }


def render_and_save(cfg: MapConfig, data: Dict[str, object]) -> str:
    """Відмальовує карту та зберігає її."""
    os.makedirs(cfg.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, cfg.fig_height_in), dpi=cfg.dpi)
    _draw_grid(ax, cfg)  # сітка

    _draw_ellipse(ax, data["ellipse"], color="black")  # еліпс

    # Точки та кола
    _draw_points(ax, data["ops_base"], label="ОП (базові)", size=12)
    _draw_points(ax, data["ops_rop"], label="ОП (РОП)", size=12, alpha=0.9)
    _draw_circles(ax, data["rop_circles"], lw=1.2, alpha=0.6, label="РОП-кола")
    _draw_circles(ax, data["op_circles"], lw=0.7, alpha=0.35, label="ОП-кола")

    _draw_battle_line(ax, data["battle_line"])  # лінія бою

    # Підписи точок старту/цілі
    ax.scatter([cfg.start[0]], [cfg.start[1]], s=40, marker="^", label="Старт")
    ax.scatter([cfg.goal[0]], [cfg.goal[1]], s=40, marker="*", label="Ціль")

    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_title(
        f"Карта з ОП у похиленому еліпсі (a={data['ellipse'].a:.1f} км, b={data['ellipse'].b:.1f} км, "
        f"кут={data['ellipse'].angle_deg:.1f}°)"
    )
    ax.margins(0.01)

    out_path = os.path.join(cfg.out_dir, "map.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_all_to_csv(cfg: MapConfig, data: Dict[str, object]) -> Dict[str, str]:
    """Експортує всі елементи карти в CSV файли."""
    os.makedirs(cfg.out_dir, exist_ok=True)

    paths = {}
    paths["ops_base_csv"] = os.path.join(cfg.out_dir, "ops_base.csv")
    paths["ops_rop_csv"] = os.path.join(cfg.out_dir, "ops_rop.csv")
    paths["ops_all_csv"] = os.path.join(cfg.out_dir, "ops_all.csv")
    paths["rop_circles_csv"] = os.path.join(cfg.out_dir, "rop_circles.csv")
    paths["op_circles_csv"] = os.path.join(cfg.out_dir, "op_circles.csv")
    paths["battle_line_csv"] = os.path.join(cfg.out_dir, "battle_line_simple.csv")
    paths["battle_line_detailed_csv"] = os.path.join(cfg.out_dir, "battle_line_detailed.csv")

    save_ops_csv(data["ops_base"], paths["ops_base_csv"])  # збереження базових ОП
    save_ops_csv(data["ops_rop"], paths["ops_rop_csv"])  # збереження РОП
    save_ops_csv(data["ops_all"], paths["ops_all_csv"])  # збереження всіх ОП
    save_circles_csv(data["rop_circles"], paths["rop_circles_csv"])  # збереження РОП кіл
    save_circles_csv(data["op_circles"], paths["op_circles_csv"])  # збереження ОП кіл

    bl: BattleLine = data["battle_line"]
    save_battle_line_to_csv(bl, paths["battle_line_csv"])  # збереження лінії бою
    save_battle_line_detailed(bl, cfg, paths["battle_line_detailed_csv"])  # збереження деталей лінії

    return paths


def _read_csv_header(path: str) -> List[str]:
    """Читає заголовки CSV файлу."""
    with open(path, "r", encoding="utf-8") as f:
        row = f.readline().strip()
    return [c.strip('# ').strip() for c in row.split(',')]




def run_demo(cfg: Optional[MapConfig] = None, cor: Optional[CorridorConfig] = None) -> Dict[str, str]:
    """Запускає демонстрацію функціональності."""
    cfg = cfg or MapConfig()
    cor = cor or CorridorConfig()

    data = build_map(cfg, cor)  # побудова карти
    png_path = render_and_save(cfg, data)  # рендеринг
    csv_paths = export_all_to_csv(cfg, data)  # експорт
    return {"png": png_path, **csv_paths}


if __name__ == "__main__":


    paths = run_demo()  # запуск демонстрації
    print("Збережено файли:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
