# ===== file: information_reflection.py =====
from __future__ import annotations  # Додає підтримку анотацій типів

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

from map_generation import (  # Імпорт із вашого модуля map_generation
    MapConfig, Point, EllipseParams, CorridorConfig,
    compute_ellipse_params, zone_boundaries,
    generate_ops_in_zones, generate_rop_clusters, generate_op_circles,
    generate_battle_line, BattleLine, MineBarrier
)
from typing import List, Tuple, Sequence


# ============================= ВІЗУАЛІЗАЦІЯ =============================

def build_figure(cfg: MapConfig) -> Tuple[Figure, Axes]:
    """Створює фігуру та осі для малювання карти."""
    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, cfg.fig_height_in), dpi=cfg.dpi)
    ax.set_xlim(0, cfg.width_km)
    ax.set_ylim(0, cfg.height_km)
    ax.set_aspect('auto')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Кілометри (X)')
    ax.set_ylabel('Кілометри (Y)')
    return fig, ax


def draw_battle_line(ax: Axes, bl: BattleLine, color: str = 'red',
                     label: str = 'Лінія бойового зіткнення') -> None:
    """Малює лінію бойового зіткнення."""
    segments = [((bl.x[i], bl.y[i]), (bl.x[i + 1], bl.y[i + 1]))
                for i in range(len(bl.x) - 1)]
    lc = LineCollection(segments, linewidths=bl.widths[:-1] * 10,
                        colors=color, label=label)
    ax.add_collection(lc)


def get_battle_line_coords(bl: BattleLine) -> List[Tuple[float, float]]:
    """Повертає список координат ЛБЗ."""
    return list(zip(bl.x, bl.y))


def draw_avg(ax: Axes, avg_points: Sequence[Point], color: str = 'black',
             label: str = 'МВГ') -> None:
    """Малює точки МВГ."""
    if avg_points:
        xs, ys = zip(*avg_points)
        ax.plot(xs, ys, 'o', color=color, markersize=4, label=label)


def plot_scene(ax: Axes,
               cfg: MapConfig,
               start: Point,
               goal: Point,
               p: EllipseParams,
               ops: Sequence[Point],
               rop_circles: Sequence[Tuple[Point, float]],
               op_circles: Sequence[Tuple[Point, float]] | None = None,
               battle_line: BattleLine | None = None,
               avg_points: Sequence[Point] | None = None,
               title: str = 'Карта з ОП у межах похиленого еліпса з зонами') -> None:
    """Малює всю сцену."""
    ax.set_title(title)

    # ЛБЗ
    if battle_line is not None:
        draw_battle_line(ax, battle_line)

    # Старт і ціль
    ax.plot(*start, 'go', markersize=8, label='Старт')
    ax.plot(*goal, 'r*', markersize=8, label='Ціль')

    # ОП
    if ops:
        xs, ys = zip(*ops)
        ax.plot(xs, ys, 'o', color='gray', markersize=2, label='ОП')

    # Кола навколо ОП
    if op_circles:
        for center, r in op_circles:
            circle = plt.Circle(center, r, color='orange',
                                fill=False, linestyle=':', linewidth=0.8)
            ax.add_patch(circle)

    # МВГ
    if avg_points:
        draw_avg(ax, avg_points)

    ax.legend(loc='best')


# ============================= ОСНОВНИЙ СЦЕНАРІЙ =============================

def run_demo(start: Point = MapConfig.start,
             goal: Point = MapConfig.goal,
             map_cfg: MapConfig | None = None,
             corridor_cfg: CorridorConfig | None = None,
             show: bool = True) -> Tuple[Figure, Axes, List[Point]]:
    """Запускає сценарій генерації та візуалізації."""
    map_cfg = map_cfg or MapConfig()
    corridor_cfg = corridor_cfg or CorridorConfig()

    rng = np.random.default_rng(corridor_cfg.random_seed)

    # Параметри еліпса
    p = compute_ellipse_params(start, goal, corridor_cfg.b)

    # Межі зон
    z_bounds, num_zones, _ = zone_boundaries(p.a, corridor_cfg.max_zone_width)

    # Генерація ОП
    op_points = generate_ops_in_zones(p, z_bounds, rng, corridor_cfg.points_per_zone)

    # РОП + додаткові ОП
    extra_ops, circles = generate_rop_clusters(
        p, z_bounds, rng,
        corridor_cfg.rop_zone_step,
        corridor_cfg.rop_radius_range,
        corridor_cfg.rop_points_range,
        op_points,
    )
    op_points.extend(extra_ops)

    # Кола навколо ОП
    op_small_circles = generate_op_circles(op_points, rng)

    # ЛБЗ
    battle = generate_battle_line(map_cfg, seed=corridor_cfg.random_seed or 42)

    # МВГ

    # Малювання всієї сцени
    fig, ax = build_figure(map_cfg)
    plot_scene(
        ax, map_cfg, start, goal, p, op_points, circles,
        op_circles=op_small_circles,
        battle_line=battle,

        title=f'Карта з ОП (≈{num_zones} зон) + МВГ'
    )

    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig('map2.png', dpi=1000, bbox_inches='tight')

    # Вивід у термінал


    return fig, ax, op_points


if __name__ == "__main__":
    run_demo()
