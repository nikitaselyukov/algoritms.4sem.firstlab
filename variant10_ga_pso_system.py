"""
Единая программа для лабораторной работы №1.
Вариант 10: функция Эггхолдера.
ГА: модификация выбора родителя; РА: модификация инерционного веса.

Запуск GUI:
    python variant10_ga_pso_system.py

Быстрый запуск демонстрационных экспериментов без интерфейса:
    python variant10_ga_pso_system.py --demo
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import matplotlib.pyplot as plt

LOWER_BOUND = -512.0
UPPER_BOUND = 512.0
SEARCH_RANGE = UPPER_BOUND - LOWER_BOUND
TARGET_MINIMUM = -959.6407
TARGET_X = 512.0
TARGET_Y = 404.2319

Encoding = Literal["real", "binary"]
SelectionMethod = Literal["roulette", "tournament"]
PSOMode = Literal["classic", "inertia"]


def eggholder(x: float, y: float) -> float:
    """Целевая функция варианта 10. Требуется найти минимум на [-512; 512]^2."""
    return -(
        (y + 47.0) * math.sin(math.sqrt(abs(x / 2.0 + y + 47.0)))
        + x * math.sin(math.sqrt(abs(x - (y + 47.0))))
    )


def clamp(value: float, low: float = LOWER_BOUND, high: float = UPPER_BOUND) -> float:
    """Возвращает значение, не выходящее за границы области поиска."""
    return max(low, min(high, value))


def make_even(value: int) -> int:
    """ГА удобнее выполнять с четным числом особей."""
    return value if value % 2 == 0 else value + 1


@dataclass(slots=True)
class Individual:
    x: float
    y: float
    fitness: float
    x_bits: str = ""
    y_bits: str = ""


@dataclass(slots=True)
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    pbest_x: float
    pbest_y: float
    pbest_value: float


@dataclass(slots=True)
class SearchResult:
    algorithm: str
    best_x: float
    best_y: float
    best_value: float
    iterations_done: int
    evaluations: int
    history: list[float]


def bits_to_float(bits: str, bits_per_coordinate: int) -> float:
    integer_value = int(bits, 2)
    return LOWER_BOUND + integer_value * SEARCH_RANGE / (2**bits_per_coordinate - 1)


def random_bits(bits_per_coordinate: int) -> str:
    return "".join(random.choice("01") for _ in range(bits_per_coordinate))


def roulette_selection(population: list[Individual]) -> list[Individual]:
    """
    Базовый выбор родителей для задачи минимизации.
    Чем меньше fitness, тем больше вес особи в рулетке.
    """
    values = [ind.fitness for ind in population]
    worst = max(values)
    weights = [worst - value + 1e-12 for value in values]
    total = sum(weights)

    if total <= 0 or not math.isfinite(total):
        return [random.choice(population) for _ in population]

    parents: list[Individual] = []
    for _ in population:
        point = random.uniform(0.0, total)
        current = 0.0
        for ind, weight in zip(population, weights):
            current += weight
            if current >= point:
                parents.append(ind)
                break
    return parents


def tournament_selection(population: list[Individual], tournament_size: int) -> list[Individual]:
    """
    Модификация ГА по варианту: выбор родителя через турнир.
    Из случайно выбранных k кандидатов родителем становится лучший.
    """
    k = max(2, min(tournament_size, len(population)))
    parents: list[Individual] = []
    for _ in population:
        candidates = random.sample(population, k)
        parents.append(min(candidates, key=lambda ind: ind.fitness))
    return parents


def select_parents(
    population: list[Individual],
    method: SelectionMethod,
    tournament_size: int,
) -> list[Individual]:
    if method == "tournament":
        return tournament_selection(population, tournament_size)
    return roulette_selection(population)


def create_real_population(size: int, evaluate: Callable[[float, float], float]) -> list[Individual]:
    population: list[Individual] = []
    for _ in range(size):
        x = random.uniform(LOWER_BOUND, UPPER_BOUND)
        y = random.uniform(LOWER_BOUND, UPPER_BOUND)
        population.append(Individual(x=x, y=y, fitness=evaluate(x, y)))
    return population


def create_binary_population(
    size: int,
    bits_per_coordinate: int,
    evaluate: Callable[[float, float], float],
) -> list[Individual]:
    population: list[Individual] = []
    for _ in range(size):
        x_bits = random_bits(bits_per_coordinate)
        y_bits = random_bits(bits_per_coordinate)
        x = bits_to_float(x_bits, bits_per_coordinate)
        y = bits_to_float(y_bits, bits_per_coordinate)
        population.append(Individual(x=x, y=y, fitness=evaluate(x, y), x_bits=x_bits, y_bits=y_bits))
    return population


def crossover_real(parent_a: Individual, parent_b: Individual) -> tuple[tuple[float, float], tuple[float, float]]:
    alpha = random.random()
    child_1 = (
        alpha * parent_a.x + (1.0 - alpha) * parent_b.x,
        alpha * parent_a.y + (1.0 - alpha) * parent_b.y,
    )
    child_2 = (
        (1.0 - alpha) * parent_a.x + alpha * parent_b.x,
        (1.0 - alpha) * parent_a.y + alpha * parent_b.y,
    )
    return child_1, child_2


def mutate_real_points(points: list[tuple[float, float]], mutation_rate: float) -> list[tuple[float, float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    sigma_x = max(0.1 * (max(xs) - min(xs)), 1.0)
    sigma_y = max(0.1 * (max(ys) - min(ys)), 1.0)

    mutated: list[tuple[float, float]] = []
    for x, y in points:
        if random.random() < mutation_rate:
            x += random.gauss(0.0, sigma_x)
            y += random.gauss(0.0, sigma_y)
        mutated.append((clamp(x), clamp(y)))
    return mutated


def mutate_bits(bits: str, mutation_rate: float) -> str:
    changed: list[str] = []
    for bit in bits:
        if random.random() < mutation_rate:
            changed.append("1" if bit == "0" else "0")
        else:
            changed.append(bit)
    return "".join(changed)


def crossover_binary(
    parent_a: Individual,
    parent_b: Individual,
    bits_per_coordinate: int,
) -> tuple[tuple[str, str], tuple[str, str]]:
    point = random.randint(1, bits_per_coordinate - 1)
    child_1 = (
        parent_a.x_bits[:point] + parent_b.x_bits[point:],
        parent_a.y_bits[:point] + parent_b.y_bits[point:],
    )
    child_2 = (
        parent_b.x_bits[:point] + parent_a.x_bits[point:],
        parent_b.y_bits[:point] + parent_a.y_bits[point:],
    )
    return child_1, child_2


def build_children(
    parents: list[Individual],
    encoding: Encoding,
    mutation_rate: float,
    bits_per_coordinate: int,
    evaluate: Callable[[float, float], float],
) -> list[Individual]:
    children: list[Individual] = []

    for index in range(0, len(parents), 2):
        parent_a = parents[index]
        parent_b = parents[index + 1] if index + 1 < len(parents) else parents[0]

        if encoding == "real":
            points = list(crossover_real(parent_a, parent_b))
            for x, y in mutate_real_points(points, mutation_rate):
                children.append(Individual(x=x, y=y, fitness=evaluate(x, y)))
        else:
            bit_children = crossover_binary(parent_a, parent_b, bits_per_coordinate)
            for x_bits, y_bits in bit_children:
                x_bits = mutate_bits(x_bits, mutation_rate)
                y_bits = mutate_bits(y_bits, mutation_rate)
                x = bits_to_float(x_bits, bits_per_coordinate)
                y = bits_to_float(y_bits, bits_per_coordinate)
                children.append(Individual(x=x, y=y, fitness=evaluate(x, y), x_bits=x_bits, y_bits=y_bits))

    return children


def next_generation(population: list[Individual], children: list[Individual]) -> list[Individual]:
    """Элитизм: лучшая найденная особь не теряется при переходе к новому поколению."""
    elite = min(population, key=lambda ind: ind.fitness)
    children.sort(key=lambda ind: ind.fitness)
    return [elite] + children[: len(population) - 1]


def genetic_algorithm(
    *,
    encoding: Encoding = "real",
    selection: SelectionMethod = "tournament",
    population_size: int = 80,
    generations: int = 80,
    mutation_rate: float = 0.15,
    tournament_size: int = 5,
    bits_per_coordinate: int = 20,
    target_tolerance: float | None = None,
    seed: int | None = None,
) -> SearchResult:
    """Запускает ГА с вещественным или бинарным кодированием."""
    if seed is not None:
        random.seed(seed)

    population_size = make_even(max(4, population_size))
    mutation_rate = max(0.0, min(1.0, mutation_rate))
    evaluations = 0

    def evaluate(x: float, y: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return eggholder(x, y)

    if encoding == "binary":
        population = create_binary_population(population_size, bits_per_coordinate, evaluate)
    else:
        population = create_real_population(population_size, evaluate)

    history: list[float] = []
    iterations_done = generations

    for generation in range(generations):
        parents = select_parents(population, selection, tournament_size)
        children = build_children(parents, encoding, mutation_rate, bits_per_coordinate, evaluate)
        population = next_generation(population, children)
        best = min(population, key=lambda ind: ind.fitness)
        history.append(best.fitness)

        if target_tolerance is not None and abs(best.fitness - TARGET_MINIMUM) <= target_tolerance:
            iterations_done = generation + 1
            break

    best = min(population, key=lambda ind: ind.fitness)
    label = f"ГА {'бинарный' if encoding == 'binary' else 'вещественный'}, {'турнир' if selection == 'tournament' else 'рулетка'}"
    return SearchResult(label, best.x, best.y, best.fitness, iterations_done, evaluations, history)


def particle_swarm(
    *,
    mode: PSOMode = "inertia",
    particles_count: int = 100,
    iterations: int = 80,
    c1: float = 1.7,
    c2: float = 1.7,
    w_max: float = 0.9,
    w_min: float = 0.4,
    velocity_limit: float | None = None,
    target_tolerance: float | None = None,
    seed: int | None = None,
) -> SearchResult:
    """Запускает классический РА или РА с линейно убывающим инерционным весом."""
    if seed is not None:
        random.seed(seed)

    particles_count = max(2, particles_count)
    evaluations = 0

    def evaluate(x: float, y: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return eggholder(x, y)

    particles: list[Particle] = []
    gbest_x = 0.0
    gbest_y = 0.0
    gbest_value = float("inf")

    for _ in range(particles_count):
        x = random.uniform(LOWER_BOUND, UPPER_BOUND)
        y = random.uniform(LOWER_BOUND, UPPER_BOUND)
        vx = random.uniform(-100.0, 100.0)
        vy = random.uniform(-100.0, 100.0)
        value = evaluate(x, y)
        particles.append(Particle(x, y, vx, vy, x, y, value))
        if value < gbest_value:
            gbest_x, gbest_y, gbest_value = x, y, value

    history: list[float] = []
    iterations_done = iterations

    for iteration in range(iterations):
        if mode == "inertia":
            inertia = w_max - (w_max - w_min) * iteration / max(1, iterations - 1)
        else:
            inertia = 1.0

        for particle in particles:
            # Независимые случайные коэффициенты для каждой координаты и компоненты
            r1x, r1y = random.random(), random.random()
            r2x, r2y = random.random(), random.random()
            particle.vx = (
                inertia * particle.vx
                + c1 * r1x * (particle.pbest_x - particle.x)
                + c2 * r2x * (gbest_x - particle.x)
            )
            particle.vy = (
                inertia * particle.vy
                + c1 * r1y * (particle.pbest_y - particle.y)
                + c2 * r2y * (gbest_y - particle.y)
            )

            if velocity_limit is not None:
                particle.vx = clamp(particle.vx, -velocity_limit, velocity_limit)
                particle.vy = clamp(particle.vy, -velocity_limit, velocity_limit)

            new_x = particle.x + particle.vx
            new_y = particle.y + particle.vy
            # При выходе за границу — отражение скорости, чтобы не прилипать к стенкам
            if new_x < LOWER_BOUND or new_x > UPPER_BOUND:
                particle.vx *= -0.5
            if new_y < LOWER_BOUND or new_y > UPPER_BOUND:
                particle.vy *= -0.5
            particle.x = clamp(new_x)
            particle.y = clamp(new_y)
            value = evaluate(particle.x, particle.y)

            if value < particle.pbest_value:
                particle.pbest_x = particle.x
                particle.pbest_y = particle.y
                particle.pbest_value = value

            if particle.pbest_value < gbest_value:
                gbest_x = particle.pbest_x
                gbest_y = particle.pbest_y
                gbest_value = particle.pbest_value

        history.append(gbest_value)
        if target_tolerance is not None and abs(gbest_value - TARGET_MINIMUM) <= target_tolerance:
            iterations_done = iteration + 1
            break

    label = "РА с инерцией" if mode == "inertia" else "РА классический"
    return SearchResult(label, gbest_x, gbest_y, gbest_value, iterations_done, evaluations, history)


def average_runs(factory: Callable[[int], SearchResult], runs: int) -> dict[str, float | str]:
    results = [factory(seed) for seed in range(runs)]
    return {
        "algorithm": results[0].algorithm,
        "runs": runs,
        "mean_best": statistics.mean(result.best_value for result in results),
        "best_value": min(result.best_value for result in results),
        "mean_iterations": statistics.mean(result.iterations_done for result in results),
        "mean_evaluations": statistics.mean(result.evaluations for result in results),
    }


def save_history_plot(results: Iterable[SearchResult], title: str, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for result in results:
        plt.plot(result.history, label=result.algorithm)
    plt.axhline(TARGET_MINIMUM, linestyle="--", label="известный минимум")
    plt.xlabel("Итерация / поколение")
    plt.ylabel("Лучшее значение функции")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_demo_experiments(output_dir: str | Path = "variant10_results", runs: int = 10) -> list[dict[str, float | str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    single_results = [
        genetic_algorithm(encoding="real", selection="roulette", population_size=100, generations=100, mutation_rate=0.2, seed=1),
        genetic_algorithm(encoding="real", selection="tournament", population_size=100, generations=100, mutation_rate=0.2, tournament_size=5, seed=1),
        genetic_algorithm(encoding="binary", selection="roulette", population_size=100, generations=100, mutation_rate=0.05, seed=1),
        genetic_algorithm(encoding="binary", selection="tournament", population_size=100, generations=100, mutation_rate=0.05, tournament_size=5, seed=1),
        particle_swarm(mode="classic", particles_count=80, iterations=100, c1=1.7, c2=1.7, seed=1),
        particle_swarm(mode="inertia", particles_count=80, iterations=100, c1=1.7, c2=1.7, seed=1),
    ]
    save_history_plot(single_results[:4], "Сходимость вариантов генетического алгоритма", output_path / "ga_convergence.png")
    save_history_plot(single_results[4:], "Сходимость роевого алгоритма", output_path / "pso_convergence.png")
    save_history_plot(single_results, "Общее сравнение алгоритмов", output_path / "all_convergence.png")

    table = [
        average_runs(
            lambda seed: genetic_algorithm(encoding="real", selection="roulette", population_size=100, generations=100, mutation_rate=0.2, seed=seed),
            runs,
        ),
        average_runs(
            lambda seed: genetic_algorithm(encoding="real", selection="tournament", population_size=100, generations=100, mutation_rate=0.2, tournament_size=5, seed=seed),
            runs,
        ),
        average_runs(
            lambda seed: genetic_algorithm(encoding="binary", selection="roulette", population_size=100, generations=100, mutation_rate=0.05, seed=seed),
            runs,
        ),
        average_runs(
            lambda seed: genetic_algorithm(encoding="binary", selection="tournament", population_size=100, generations=100, mutation_rate=0.05, tournament_size=5, seed=seed),
            runs,
        ),
        average_runs(
            lambda seed: particle_swarm(mode="classic", particles_count=80, iterations=100, c1=1.7, c2=1.7, seed=seed),
            runs,
        ),
        average_runs(
            lambda seed: particle_swarm(mode="inertia", particles_count=80, iterations=100, c1=1.7, c2=1.7, seed=seed),
            runs,
        ),
    ]

    with open(output_path / "summary.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(table)

    return table


def print_demo_table(table: list[dict[str, float | str]]) -> None:
    print("Алгоритм;Запусков;Средний минимум;Лучший минимум;Среднее число итераций;Среднее число вычислений")
    for row in table:
        print(
            f"{row['algorithm']};{row['runs']};{row['mean_best']:.4f};{row['best_value']:.4f};"
            f"{row['mean_iterations']:.1f};{row['mean_evaluations']:.1f}"
        )


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Лабораторная 1: ГА и РА для варианта 10")
    root.geometry("650x520")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    def add_entry(parent: tk.Widget, label: str, default: str, row: int) -> tk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=4)
        entry = ttk.Entry(parent, width=18)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=4)
        return entry

    ga_tab = ttk.Frame(notebook)
    notebook.add(ga_tab, text="Генетический алгоритм")

    ga_pop = add_entry(ga_tab, "Размер популяции", "120", 0)
    ga_gen = add_entry(ga_tab, "Количество поколений", "80", 1)
    ga_mut = add_entry(ga_tab, "Вероятность мутации", "0.2", 2)
    ga_tour = add_entry(ga_tab, "Размер турнира", "5", 3)
    ga_bits = add_entry(ga_tab, "Битов на координату", "20", 4)
    ga_tol = add_entry(ga_tab, "Точность остановки (пусто = без нее)", "", 5)

    ttk.Label(ga_tab, text="Кодирование").grid(row=6, column=0, sticky="w", padx=5, pady=4)
    ga_encoding = tk.StringVar(value="real")
    ttk.Combobox(ga_tab, textvariable=ga_encoding, values=["real", "binary"], state="readonly", width=16).grid(row=6, column=1, sticky="w", padx=5, pady=4)

    ttk.Label(ga_tab, text="Выбор родителей").grid(row=7, column=0, sticky="w", padx=5, pady=4)
    ga_selection = tk.StringVar(value="tournament")
    ttk.Combobox(ga_tab, textvariable=ga_selection, values=["roulette", "tournament"], state="readonly", width=16).grid(row=7, column=1, sticky="w", padx=5, pady=4)

    ga_result = ttk.Label(ga_tab, text="Результат появится после запуска", justify="left")
    ga_result.grid(row=9, column=0, columnspan=2, sticky="w", padx=5, pady=10)

    def run_ga_from_gui() -> None:
        try:
            tolerance_text = ga_tol.get().strip()
            tolerance = float(tolerance_text) if tolerance_text else None
            result = genetic_algorithm(
                encoding=ga_encoding.get(),
                selection=ga_selection.get(),
                population_size=int(ga_pop.get()),
                generations=int(ga_gen.get()),
                mutation_rate=float(ga_mut.get()),
                tournament_size=int(ga_tour.get()),
                bits_per_coordinate=int(ga_bits.get()),
                target_tolerance=tolerance,
            )
            ga_result.config(
                text=(
                    f"{result.algorithm}\n"
                    f"Лучшее значение: {result.best_value:.6f}\n"
                    f"x = {result.best_x:.6f}; y = {result.best_y:.6f}\n"
                    f"Поколений выполнено: {result.iterations_done}\n"
                    f"Вычислений функции: {result.evaluations}"
                )
            )
            save_history_plot([result], result.algorithm, Path("ga_last_run.png"))
            plt.figure(figsize=(8, 4))
            plt.plot(result.history, label=result.algorithm)
            plt.axhline(TARGET_MINIMUM, linestyle="--", label="известный минимум")
            plt.xlabel("Поколение")
            plt.ylabel("Лучшее значение")
            plt.title(result.algorithm)
            plt.grid(True)
            plt.legend()
            plt.show()
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте, что все параметры введены числами.")

    ttk.Button(ga_tab, text="Запустить ГА", command=run_ga_from_gui).grid(row=8, column=0, columnspan=2, pady=8)

    pso_tab = ttk.Frame(notebook)
    notebook.add(pso_tab, text="Роевой алгоритм")

    pso_n = add_entry(pso_tab, "Количество частиц", "120", 0)
    pso_iter = add_entry(pso_tab, "Количество итераций", "80", 1)
    pso_c1 = add_entry(pso_tab, "Коэффициент c1", "1.7", 2)
    pso_c2 = add_entry(pso_tab, "Коэффициент c2", "1.7", 3)
    pso_wmax = add_entry(pso_tab, "w max", "0.9", 4)
    pso_wmin = add_entry(pso_tab, "w min", "0.4", 5)
    pso_tol = add_entry(pso_tab, "Точность остановки (пусто = без нее)", "", 6)

    ttk.Label(pso_tab, text="Режим").grid(row=7, column=0, sticky="w", padx=5, pady=4)
    pso_mode = tk.StringVar(value="inertia")
    ttk.Combobox(pso_tab, textvariable=pso_mode, values=["classic", "inertia"], state="readonly", width=16).grid(row=7, column=1, sticky="w", padx=5, pady=4)

    pso_result = ttk.Label(pso_tab, text="Результат появится после запуска", justify="left")
    pso_result.grid(row=9, column=0, columnspan=2, sticky="w", padx=5, pady=10)

    def run_pso_from_gui() -> None:
        try:
            tolerance_text = pso_tol.get().strip()
            tolerance = float(tolerance_text) if tolerance_text else None
            result = particle_swarm(
                mode=pso_mode.get(),
                particles_count=int(pso_n.get()),
                iterations=int(pso_iter.get()),
                c1=float(pso_c1.get()),
                c2=float(pso_c2.get()),
                w_max=float(pso_wmax.get()),
                w_min=float(pso_wmin.get()),
                target_tolerance=tolerance,
            )
            pso_result.config(
                text=(
                    f"{result.algorithm}\n"
                    f"Лучшее значение: {result.best_value:.6f}\n"
                    f"x = {result.best_x:.6f}; y = {result.best_y:.6f}\n"
                    f"Итераций выполнено: {result.iterations_done}\n"
                    f"Вычислений функции: {result.evaluations}"
                )
            )
            save_history_plot([result], result.algorithm, Path("pso_last_run.png"))
            plt.figure(figsize=(8, 4))
            plt.plot(result.history, label=result.algorithm)
            plt.axhline(TARGET_MINIMUM, linestyle="--", label="известный минимум")
            plt.xlabel("Итерация")
            plt.ylabel("Лучшее значение")
            plt.title(result.algorithm)
            plt.grid(True)
            plt.legend()
            plt.show()
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте, что все параметры введены числами.")

    ttk.Button(pso_tab, text="Запустить РА", command=run_pso_from_gui).grid(row=8, column=0, columnspan=2, pady=8)

    compare_tab = ttk.Frame(notebook)
    notebook.add(compare_tab, text="Быстрое сравнение")
    compare_result = tk.Text(compare_tab, height=18, width=78)
    compare_result.pack(padx=8, pady=8)

    def run_compare_from_gui() -> None:
        table = run_demo_experiments("variant10_results", runs=5)
        compare_result.delete("1.0", tk.END)
        compare_result.insert(tk.END, "Алгоритм | средний минимум | лучший минимум | средн. итераций | средн. вычислений\n")
        compare_result.insert(tk.END, "-" * 82 + "\n")
        for row in table:
            compare_result.insert(
                tk.END,
                f"{row['algorithm']}: {row['mean_best']:.4f}; best={row['best_value']:.4f}; "
                f"iter={row['mean_iterations']:.1f}; eval={row['mean_evaluations']:.1f}\n",
            )
        compare_result.insert(tk.END, "\nГрафики и summary.csv сохранены в папку variant10_results.\n")

    ttk.Button(compare_tab, text="Провести сравнение", command=run_compare_from_gui).pack(pady=5)

    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ГА и РА для функции Эггхолдера, вариант 10")
    parser.add_argument("--demo", action="store_true", help="запустить сравнение без графического интерфейса")
    parser.add_argument("--output", default="variant10_results", help="папка для результатов demo-запуска")
    parser.add_argument("--runs", type=int, default=10, help="число повторов в demo-запуске")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        table = run_demo_experiments(args.output, args.runs)
        print_demo_table(table)
    else:
        launch_gui()


if __name__ == "__main__":
    main()
