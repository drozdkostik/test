# ===== file: map_generation.py =====                       # Назва файлу
"""
Рефакторинг коду побудови карти з ОП усередині похиленого еліпса.
- Структуровано у функції та dataclass-конфіги
- Керований ГВЧ через numpy Generator
- Окремі генерації ОП по зонах та РОП-кластерів
- Мінімізація дублювання формул та обчислень
- Україномовні докстрінги
"""  # Докстрінг з описом функціоналу
from __future__ import annotations  # Імпорт для підтримки анотацій типів

from dataclasses import dataclass  # Імпорт декоратора dataclass
from typing import List, Sequence, Tuple, Optional  # Імпорт типів для анотацій
import math  # Імпорт модуля math
import numpy as np  # Імпорт numpy для роботи з масивами
from matplotlib.axes import Axes  # Імпорт класу Axes для типізації

Point = Tuple[float, float]  # Визначення типу Point як кортеж двох float
MineBarrier = Tuple[float, float]  # Визначення типу MineBarrier як кортеж координат
CircleSpec = Tuple[Point, float]  # ((x, y), r)

# =============================
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\Конфіги////////////////////////////////////////
@dataclass
class MapConfig:  # Опис конфігурації карти
    """Параметри полотна карти (в кілометрах та пікселях)."""
    height_km: float = 50  # Висота карти в км
    width_km: float = 250  # Ширина карти в км
    cell_size: float = 10  # Розмір клітинки в км
    dpi: int = 100  # Щільність пікселів
    fig_width_in: float = 10  # Ширина фігури у дюймах
    fig_height_in: float = 2.5  # Висота фігури у дюймах
    start: float = (10, 10)  # Початкова точка у координатах (x, y)
    goal: float = (245, 30)  # Кінцева точка у координатах (x, y)
    out_dir: str = "output"


@dataclass
class CorridorConfig:  # Опис конфігурації коридору який утворений еліпсом між ТС та ТЦ
    """Параметри еліптичного коридору та генерації точок."""
    b: float = 10  # Мала піввісь еліпса
    max_zone_width: float = 10  # Максимальна ширина зони
    rop_zone_step: int = 3  # Крок зон для РОП
    points_per_zone: Tuple[int, int] = (3, 5)  # Мін/макс ОП на зону
    rop_points_range: Tuple[int, int] = (5, 30)  # Мін/макс ОП у РОП-колі
    rop_radius_range: Tuple[float, float] = (1.0, 5.0)  # Діапазон радіусів РОП-кіл
    op_radius_range: Tuple[float, float] = (0.005, 0.02)  # Діапазон радіусів ОП-кіл
    random_seed: Optional[int] = None  # Зерно для генератора випадкових чисел

# =================================================================================================================
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\Геометричні утиліти//////////////////////////////////////////
# =================================================================================================================
@dataclass
class EllipseParams:  # Опис параметрів еліпса
    x_c: float  # Центр X
    y_c: float  # Центр Y
    a: float  # Велика піввісь
    b: float  # Мала піввісь
    angle_rad: float  # Кут нахилу в радіанах
    angle_deg: float  # Кут нахилу в градусах
    cos_a: float  # Косинус кута
    sin_a: float  # Синус кута
def compute_ellipse_params(start: Point, goal: Point, b: float) -> EllipseParams:  # Обчислення параметрів еліпса
    """Обчислює параметри похиленого еліпса за стартом, ціллю та заданою малою піввіссю b."""
    x_c = (start[0] + goal[0]) / 2.0  # Центр X як середнє між стартом і ціллю
    y_c = (start[1] + goal[1]) / 2.0  # Центр Y як середнє між стартом і ціллю

    a = float(np.linalg.norm(np.array(goal) - np.array(start)) / 2.0)  # Велика піввісь як половина відстані

    dx = goal[0] - start[0]  # Різниця по X
    dy = goal[1] - start[1]  # Різниця по Y
    angle_rad = math.atan2(dy, dx)  # Кут нахилу в радіанах
    angle_deg = math.degrees(angle_rad)  # Кут нахилу в градусах

    cos_a = math.cos(angle_rad)  # Косинус кута
    sin_a = math.sin(angle_rad)  # Синус кута

    return EllipseParams(x_c, y_c, a, b, angle_rad, angle_deg, cos_a, sin_a)  # Повернення параметрів еліпса


def rotate_local_to_global(x_local: np.ndarray | float,
                           y_local: np.ndarray | float,
                           p: EllipseParams) -> Tuple[
    np.ndarray, np.ndarray]:  # Перетворення локальних координат у глобальні
    """Повертає локальні координати еліпса у глобальні."""
    xg = x_local * p.cos_a - y_local * p.sin_a + p.x_c  # Обчислення глобальної X
    yg = x_local * p.sin_a + y_local * p.cos_a + p.y_c  # Обчислення глобальної Y
    return xg, yg  # Повернення глобальних координат

def rotate_global_to_local(x_global: np.ndarray | float,
                           y_global: np.ndarray | float,
                           p: EllipseParams) -> Tuple[np.ndarray, np.ndarray]:
    """Обернене перетворення: глобальні → локальні координати еліпса."""
    x_shift = x_global - p.x_c  # зміщення по x
    y_shift = y_global - p.y_c  # зміщення по y
    xl = x_shift * p.cos_a + y_shift * p.sin_a  # локальна x-координата
    yl = -x_shift * p.sin_a + y_shift * p.cos_a  # локальна y-координата
    return xl, yl


def zone_boundaries(a: float, max_zone_width: float) -> Tuple[np.ndarray, int, float]:  # Обчислення меж зон
    """Повертає (кордони зон, кількість зон, крок) у локальних координатах [-a, a]."""
    num_zones = int(np.ceil(2 * a / max_zone_width)) if max_zone_width > 0 else 1  # Кількість зон
    step = 2 * a / num_zones  # Крок між зонами
    return np.linspace(-a, a, num_zones + 1), num_zones, step  # Повернення меж, кількості, кроку

# =================================================================================================================
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\Генерація точок//////////////////////////////////////////
# =================================================================================================================
def generate_ops_in_zones(p: EllipseParams,
                          z_bounds: Sequence[float],
                          rng: np.random.Generator,
                          points_per_zone: Tuple[int, int]) -> List[Point]:  # Генерація ОП у зонах
    """Генерує опорні пункти (ОП) у кожній зоні еліпса (локальні x у межах зони)."""
    ops: List[Point] = []  # Список для збереження ОП
    for i in range(len(z_bounds) - 1):  # Прохід по кожній зоні
        x0, x1 = z_bounds[i], z_bounds[i + 1]  # Межі зони
        n = int(rng.integers(points_per_zone[0], points_per_zone[1] + 1))  # Кількість ОП у зоні
        for _ in range(n):  # Генерація кожної ОП
            x_local = rng.uniform(x0, x1)  # Випадковий x у межах зони
            # y_max із рівняння еліпса: (x/a)^2 + (y/b)^2 <= 1
            y_max = p.b * math.sqrt(max(0.0, 1.0 - (x_local / p.a) ** 2))  # Максимальний y для цього x
            y_local = rng.uniform(-y_max, y_max)  # Випадковий y у межах допустимого
            xg, yg = rotate_local_to_global(x_local, y_local, p)  # Перетворення у глобальні координати
            ops.append((float(xg), float(yg)))  # Додавання ОП у список
    return ops  # Повернення списку ОП


def _has_overlap(candidate: Point, existing: np.ndarray, radius: float) -> bool:  # Перевірка перетину точок
    """Перевірка перетину кандидата з наявними точками за евклідовою відстанню."""
    if existing.size == 0:  # Якщо немає існуючих точок
        return False  # Перетину немає
    d = np.hypot(existing[:, 0] - candidate[0], existing[:, 1] - candidate[1])  # Відстані до всіх точок
    return bool((d <= radius).any())  # Чи є точка ближче за радіус


def generate_rop_clusters(p: EllipseParams,
                          z_bounds: Sequence[float],
                          rng: np.random.Generator,
                          step: int,
                          radius_range: Tuple[float, float],
                          points_range: Tuple[int, int],
                          existing_ops: List[Point]) -> Tuple[
    List[Point], List[Tuple[Point, float]]]:  # Генерація РОП-кластерів
    """Генерує РОП-кластери (кола) та додаткові ОП в їх межах без перетину з існуючими.

    Повертає: (нові ОП, список кіл як ((x,y), r)).
    """
    new_ops: List[Point] = []  # Список нових ОП
    circles: List[Tuple[Point, float]] = []  # Список кіл РОП

    ops_arr = np.array(existing_ops, dtype=float) if existing_ops else np.empty((0, 2))  # Масив існуючих ОП

    for i in range(0, len(z_bounds) - 1, max(1, step)):  # Прохід по зонах з кроком step
        x0, x1 = z_bounds[i], z_bounds[i + 1]  # Межі зони
        # центр по x беремо посередині зони, y — усередині половини доступного y
        x_local = 0.5 * (x0 + x1)  # Центр зони по x
        y_max = p.b * math.sqrt(max(0.0, 1.0 - (x_local / p.a) ** 2))  # Максимальний y для цього x
        attempts, max_attempts = 0, 100  # Лічильник спроб
        success = False  # Прапорець успіху
        while attempts < max_attempts and not success:  # Поки не знайдено місце або не вичерпано спроби
            y_local = rng.uniform(-0.5 * y_max, 0.5 * y_max)  # Випадковий y у половині допустимого
            xg, yg = rotate_local_to_global(x_local, y_local, p)  # Перетворення у глобальні координати
            r = float(rng.uniform(radius_range[0], radius_range[1]))  # Випадковий радіус кола
            candidate = (float(xg), float(yg))  # Кандидат на центр кола
            if not _has_overlap(candidate, ops_arr, r):  # Якщо немає перетину з існуючими
                success = True  # Позначаємо успіх
                circles.append((candidate, r))  # Додаємо коло у список
                k = int(rng.integers(points_range[0], points_range[1] + 1))  # Кількість ОП у колі
                # точки усередині кола у глобальних координатах
                angles = rng.uniform(0.0, 2 * math.pi, size=k)  # Випадкові кути для точок
                radii = rng.uniform(0.0, r, size=k)  # Випадкові радіуси для точок
                xs = candidate[0] + radii * np.cos(angles)  # X координати точок
                ys = candidate[1] + radii * np.sin(angles)  # Y координати точок
                pts = list(zip(xs.astype(float), ys.astype(float)))  # Список точок
                new_ops.extend(pts)  # Додаємо точки у список нових ОП
                # оновлюємо масив для наступних перевірок
                if ops_arr.size:
                    ops_arr = np.vstack([ops_arr, np.column_stack([xs, ys])])  # Додаємо точки до масиву
                else:
                    ops_arr = np.column_stack([xs, ys])  # Створюємо масив з точок
            attempts += 1  # Збільшуємо лічильник спроб
    return new_ops, circles  # Повертаємо нові ОП та кола


def generate_op_circles(ops: List[Point], rng: np.random.Generator) -> List[
    Tuple[Point, float]]:  # Генерація кіл навколо ОП
    """Генерує випадкові кола навколо кожного ОП з радіусом від 0.005 до 0.02 км."""
    circles: List[Tuple[Point, float]] = []  # Список кіл
    for op in ops:  # Для кожної ОП
        radius = float(rng.uniform(0.005, 0.02))  # Випадковий радіус
        circles.append((op, radius))  # Додаємо коло у список
    return circles  # Повертаємо список кіл

# =================================================================================================================
# \\\\\\\\\\\\\\\\\\\\\\\\Формування лінії бойового зіткнення (без відмалювання)////////////////////////////////////////
# =================================================================================================================
@dataclass
class BattleLine:  # Опис структури лінії бою
    x: np.ndarray  # Масив X-координат
    y: np.ndarray  # Масив Y-координат
    widths: np.ndarray  # Масив товщин лінії


def generate_battle_line(cfg: MapConfig, seed: int = 42) -> BattleLine:  # Генерація лінії бою
    """Формує дані лінії бойового зіткнення як плавну вигнуту криву в межах карти.

    Повертає BattleLine(x, y, widths), що може бути відображена у модулі візуалізації.
    """
    num_points = 200  # Кількість точок лінії
    x_center = cfg.width_km / 2.0 + 5.0  # Центр лінії по X
    y_vals = np.linspace(0, cfg.height_km, num_points)  # Масив Y-координат
    rng = np.random.default_rng(seed)  # Генератор випадкових чисел

    # Початок (y=0) -2 км від центру, кінець (y=макс) +5 км від центру
    x_start = x_center - 2.0  # Початковий X
    x_end = x_center + 3.0  # Кінцевий X
    x_base = np.linspace(x_start, x_end, num_points)  # Базовий масив X

    # Генеруємо шум і згладжуємо його згорткою для плавності
    raw_noise = rng.normal(0, 1.1, size=num_points)  # Випадковий шум
    window_size = 30  # Розмір вікна згладжування
    window = np.hanning(window_size)  # Вікно Хеннінга
    window /= window.sum()  # Нормалізація вікна
    smooth_noise = np.convolve(raw_noise, window, mode='same')  # Згладжування шуму
    smooth_noise -= np.mean(smooth_noise)  # Центрування шуму

    # Збільшуємо амплітуду вигинів до ~2 км у глибину (по X)
    smooth_noise *= 5.0  # Масштабування шуму

    # Остаточні X зі зміщенням
    x_vals = x_base + smooth_noise  # Додавання шуму до базових X
 # Випадкова ширина лінії в межах 0.05-0.1 км (масштабується під linewidth під час відмалювання)
    widths = rng.uniform(0.05, 0.1, size=num_points)  # Випадкові товщини лінії
    return BattleLine(x=x_vals, y=y_vals, widths=widths)  # Повернення структури BattleLine
# =================================================================================================================
# \\\\\\\\\\\\\\\\\\\\\\\\Генерація МВГ по ЛБЗ та по границі РОП////////////////////////////////////////




    # Генерація МВГ біля РОП
    # mvg_near_rops = generate_MVG_near_ROPs(rop_circles, battle_line)
    # print(f"[INFO] Всього згенеровано МВГ біля РОП: {len(mvg_near_rops)}")

    # Об'єднання всіх МВГ
    # all_mvg = mvg_barriers + mvg_near_rops
    # print(f"[INFO] Загальна кількість МВГ: {len(all_mvg)}")

    # Тут можна додати код для візуалізації або збереження результатів














