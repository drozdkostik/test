# ===== file: information reflection.py =====
from __future__ import annotations  # Додає підтримку анотацій типів у майбутніх версіях Python

import numpy as np  # Імпортує бібліотеку numpy для роботи з масивами та генераторами випадкових чисел
from matplotlib.axes import Axes  # Імпортує клас Axes для типізації
from matplotlib.figure import Figure  # Імпортує клас Figure для типізації
from matplotlib.collections import LineCollection  # Імпортує LineCollection для малювання ліній

from map_generation import (  # Імпортує функції та класи з модуля map_generation
    MapConfig, plt, Point, EllipseParams, CorridorConfig,
    compute_ellipse_params, zone_boundaries,
    generate_ops_in_zones, generate_rop_clusters, generate_op_circles,
    generate_battle_line, BattleLine,
)
from typing import List, Tuple, Sequence  # Імпортує типи для анотацій


# =============================
# Візуалізація
# =============================

def build_figure(cfg: MapConfig) -> Tuple[Figure, Axes]:  # Створює фігуру та осі для малювання
    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, cfg.fig_height_in), dpi=cfg.dpi)  # Створює фігуру з заданими розмірами
    ax.set_xlim(0, cfg.width_km)  # Встановлює межі осі X
    ax.set_ylim(0, cfg.height_km)  # Встановлює межі осі Y
    ax.set_aspect('auto')  # Встановлює автоматичне співвідношення сторін
    ax.grid(True, linestyle='--', linewidth=0.5)  # Додає сітку
    ax.set_xlabel('Кілометри (X)')  # Підписує вісь X
    ax.set_ylabel('Кілометри (Y)')  # Підписує вісь Y
    return fig, ax  # Повертає фігуру та осі


def draw_battle_line(ax: Axes, bl: BattleLine, color: str = 'red', label: str = 'Лінія бойового зіткнення') -> None:  # Малює лінію бойового зіткнення
    """Відображає лінію бойового зіткнення на осі matplotlib за підготовленими даними."""
    segments = [((bl.x[i], bl.y[i]), (bl.x[i + 1], bl.y[i + 1])) for i in range(len(bl.x) - 1)]  # Формує сегменти лінії
    lc = LineCollection(segments, linewidths=bl.widths[:-1] * 10, colors=color, label=label)  # Створює колекцію ліній
    ax.add_collection(lc)  # Додає лінії на осі


def plot_scene(ax: Axes,  # Відображає всю сцену на осі
               cfg: MapConfig,
               start: Point,
               goal: Point,
               p: EllipseParams,
               ops: Sequence[Point],
               rop_circles: Sequence[Tuple[Point, float]],
               op_circles: Sequence[Tuple[Point, float]] | None = None,
               battle_line: BattleLine | None = None,
               title: str = 'Карта з ОП у межах похиленого еліпса з зонами') -> None:
    ax.set_title(title)  # Встановлює заголовок

    # Лінія бойового зіткнення (спочатку, щоб легенда включала її)
    if battle_line is not None:  # Якщо передано лінію бойового зіткнення
        draw_battle_line(ax, battle_line)  # Малює лінію бойового зіткнення

    # Точки старту і цілі
    ax.plot(*start, 'go', markersize=8, label='Старт')  # Малює точку старту
    ax.plot(*goal, 'r*', markersize=8, label='Ціль')  # Малює точку цілі

    # ОП
    if ops:  # Якщо є опорні пункти
        xs, ys = zip(*ops)  # Розпаковує координати
    ax.plot(xs, ys, 'o', color='gray', markersize=2, label='ОП')  # Малює ОП сірим кольором

    # Межі еліпса (опційно — закоментовано)
    # from matplotlib.patches import Ellipse as MplEllipse  # Імпортує Ellipse для малювання еліпса
    # ellipse = MplEllipse((p.x_c, p.y_c), width=2 * p.a, height=2 * p.b,
    #                      angle=p.angle_deg, edgecolor='purple', facecolor='none',
    #                      linewidth=0.5, linestyle='--', label='Межі еліпса')  # Створює еліпс
    # ax.add_patch(ellipse)  # Додає еліпс на осі

    # РОП-кластери (кола)
    # for center, r in rop_circles:  # Для кожного РОП-кластера
    #     circle = plt.Circle(center, r, color='green', fill=False, linestyle='--', linewidth=1)  # Створює коло
    #     ax.add_patch(circle)  # Додає коло на осі

    # Кола навколо ОП (якщо передані)
    if op_circles:  # Якщо передані кола навколо ОП
        for center, r in op_circles:  # Для кожного кола
            circle = plt.Circle(center, r, color='orange', fill=False, linestyle=':', linewidth=0.8)  # Створює коло
            ax.add_patch(circle)  # Додає коло на осі

    ax.legend(loc='best')  # Додає легенду


def plot_op_circles(ax: Axes, circles: List[Tuple[Point, float]]):  # Малює кола навколо ОП
    """Відображає кола на вказаній осі matplotlib."""
    for center, radius in circles:  # Для кожного кола
        ax.add_patch(plt.Circle(center, radius, color='orange', fill=False, linestyle='--', linewidth=0.5))  # Додає коло


# =============================
# Основний сценарій (приклад використання)
# =============================

def run_demo(start: Point = MapConfig.start,  # Основна функція запуску сценарію
             goal: Point = MapConfig.goal,
             map_cfg: MapConfig | None = None,
             corridor_cfg: CorridorConfig | None = None,
             show: bool = True) -> Tuple[Figure, Axes, List[Point]]:
    """Запускає повний сценарій генерації та відмалювання. Повертає (fig, ax, op_points)."""
    map_cfg = map_cfg or MapConfig()  # Використовує передану або створює нову MapConfig
    corridor_cfg = corridor_cfg or CorridorConfig()  # Використовує передану або створює нову CorridorConfig

    rng = np.random.default_rng(corridor_cfg.random_seed)  # Створює генератор випадкових чисел

    # Параметри еліпса
    p = compute_ellipse_params(start, goal, corridor_cfg.b)  # Обчислює параметри еліпса

    # Зони та їхні кордони (локальні)
    z_bounds, num_zones, _ = zone_boundaries(p.a, corridor_cfg.max_zone_width)  # Обчислює межі зон

    # Генерація ОП у  зонах
    op_points = generate_ops_in_zones(p, z_bounds, rng, corridor_cfg.points_per_zone)  # Генерує ОП у зонах

    # РОП-кластери + додаткові ОП
    extra_ops, circles = generate_rop_clusters(  # Генерує РОП-кластери та додаткові ОП
        p,
        z_bounds,
        rng,
        corridor_cfg.rop_zone_step,
        corridor_cfg.rop_radius_range,
        corridor_cfg.rop_points_range,
        op_points,
    )
    op_points.extend(extra_ops)  # Додає додаткові ОП до списку

    # Кола навколо кожного ОП (невеликі радіуси)
    op_small_circles = generate_op_circles(op_points, rng)  # Генерує кола навколо ОП

    # Лінія бойового зіткнення — ФОРМУЄМО у MapGeneration, ВІДОБРАЖАЄМО тут
    battle = generate_battle_line(map_cfg, seed=corridor_cfg.random_seed or 42)  # Генерує лінію бойового зіткнення

    # Постановка фігури
    fig, ax = build_figure(map_cfg)  # Створює фігуру та осі
    plot_scene(
        ax, map_cfg, start, goal, p, op_points, circles,
        op_circles=op_small_circles,
        battle_line=battle,
        title=f'Карта з ОП (≈{num_zones} зон)'
    )  # Малює всю сцену

    plt.tight_layout()  # Робить макет компактним
    if show:  # Якщо потрібно показати
        plt.show()  # Відображає фігуру
    plt.savefig('map2.png', dpi=1000, bbox_inches='tight')

    # Вивід координат ОП
    print("\nСформовані координати опорних пунктів (ОП):")  # Виводить заголовок
    for idx, (x, y) in enumerate(op_points, 1):  # Для кожного ОП
        print(f"{idx:>3}: ({x:.2f}, {y:.2f})")  # Виводить координати ОП
    
    return fig, ax, op_points  # Повертає фігуру, осі та список ОП



if __name__ == "__main__":  # Якщо файл запущено напряму
    run_demo()  # Запускає демонстраційний сценарій
