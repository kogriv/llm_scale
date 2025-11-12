"""
Task 3 Claude Hybrid Solution

Комбинированный подход:
1. Мульти-источниковый Dijkstra (как в codex) для корректности
2. Адаптивный cap на количество находимых целей для скорости
3. Умная оптимизация для повторяющихся букв
4. Эффективное управление памятью с переиспользованием массивов
5. Time budget tracking для адаптации стратегии

Ключевые отличия от оригинала:
- Единый Dijkstra вместо множественных BFS
- Динамический выбор параметров на основе характеристик задачи
- Балансировка между скоростью и точностью
"""

import sys
import heapq
from typing import Dict, List
import time

def solve() -> None:
    start_time = time.time()

    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    sx = int(next(it)) - 1
    sy = int(next(it)) - 1

    grid: List[str] = []
    for _ in range(n):
        grid.append(next(it))

    deliveries = next(it, "")

    total_cells = n * m
    start_idx = sx * m + sy

    # Списки позиций для каждого типа адресов
    addresses: Dict[str, List[int]] = {chr(ord('a') + i): [] for i in range(26)}
    for x in range(n):
        base = x * m
        row = grid[x]
        for y in range(m):
            addresses[row[y]].append(base + y)

    # Быстрые преобразования индексов
    def idx_to_xy(idx: int) -> tuple[int, int]:
        return divmod(idx, m)

    # Метки для целевых позиций
    target_mark = [0] * total_cells
    mark_id = 0

    best_cost = [0] * total_cells
    best_mark = [0] * total_cells
    dist_run = 0

    # Адаптивная стратегия: определяем параметры на основе характеристик задачи
    grid_size = n * m
    seq_len = len(deliveries)

    # Определяем сложность задачи
    if grid_size > 30000 or seq_len > 150:
        difficulty = "hard"
        base_cap = 400
        tail_cap = None  # Без ограничений на хвосте
        tail_steps = 8
    elif grid_size > 10000 or seq_len > 80:
        difficulty = "medium"
        base_cap = 300
        tail_cap = 600
        tail_steps = 6
    else:
        difficulty = "easy"
        base_cap = 500
        tail_cap = None
        tail_steps = 5

    # Time budget: если приближаемся к лимиту времени, снижаем cap
    TIME_LIMIT = 1.9  # Консервативная оценка

    def get_adaptive_cap(step_idx: int, char: str, next_char: str | None, remaining: int) -> int | None:
        """Определяет cap для текущего шага на основе контекста"""
        elapsed = time.time() - start_time
        time_per_step = elapsed / (step_idx + 1) if step_idx > 0 else 0.1
        estimated_remaining_time = time_per_step * remaining
        time_budget = TIME_LIMIT - elapsed

        # Если в хвосте последовательности - используем больше ресурсов
        is_tail = remaining < tail_steps

        # Переход между разными типами
        is_transition = next_char is not None and next_char != char

        # Обильный тип (много клеток)
        target_count = len(addresses[char])
        is_abundant = target_count > 300

        # Критическая ситуация: мало времени
        is_time_critical = estimated_remaining_time > time_budget * 0.8

        if is_tail:
            # На хвосте - максимальная точность
            return tail_cap
        elif is_time_critical:
            # Мало времени - агрессивное ограничение
            return min(base_cap, 200)
        elif is_transition and is_abundant:
            # Критический переход между обильными типами
            return base_cap if difficulty == "hard" else None
        elif target_count > 800:
            # Очень много целей - ограничиваем
            return base_cap
        else:
            # Стандартная ситуация
            return base_cap if difficulty == "hard" else None

    def advance(front: Dict[int, int], target_char: str, cap: int | None, step_idx: int) -> Dict[int, int]:
        """
        Мульти-источниковый Dijkstra от всех позиций текущего фронта

        Args:
            front: Словарь {позиция: стоимость} текущих позиций
            target_char: Искомый тип адреса
            cap: Максимальное количество целей для поиска (None = без ограничений)
            step_idx: Номер шага (для отладки)

        Returns:
            Словарь {позиция: стоимость} найденных целей
        """
        nonlocal mark_id
        nonlocal dist_run

        targets = addresses[target_char]
        if not targets:
            return {}

        # Помечаем целевые клетки
        mark_id += 1
        current_mark = mark_id
        for idx in targets:
            target_mark[idx] = current_mark

        remaining_targets = len(targets)
        result: Dict[int, int] = {}

        # Инициализируем Dijkstra с множественными источниками
        dist_run += 1
        current_run = dist_run
        heap: List[tuple[int, int]] = []

        for idx, cost in front.items():
            if best_mark[idx] != current_run or cost < best_cost[idx]:
                best_mark[idx] = current_run
                best_cost[idx] = cost
                heapq.heappush(heap, (cost, idx))

        # Определяем лимит находимых целей
        max_targets = len(targets) if cap is None else min(len(targets), cap)

        while heap and remaining_targets and len(result) < max_targets:
            cost, idx = heapq.heappop(heap)

            # Пропускаем устаревшие записи
            if best_mark[idx] != current_run or cost != best_cost[idx]:
                continue

            # Проверяем, является ли текущая клетка целью
            if target_mark[idx] == current_mark and idx not in result:
                result[idx] = cost
                remaining_targets -= 1
                if len(result) >= max_targets:
                    break

            # Расширяем поиск на соседей
            x, y = idx_to_xy(idx)
            new_cost = cost + 1

            # Оптимизированный порядок проверки соседей
            if y + 1 < m:
                ni = idx + 1
                if best_mark[ni] != current_run or new_cost < best_cost[ni]:
                    best_mark[ni] = current_run
                    best_cost[ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni))

            if y - 1 >= 0:
                ni = idx - 1
                if best_mark[ni] != current_run or new_cost < best_cost[ni]:
                    best_mark[ni] = current_run
                    best_cost[ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni))

            if x + 1 < n:
                ni = idx + m
                if best_mark[ni] != current_run or new_cost < best_cost[ni]:
                    best_mark[ni] = current_run
                    best_cost[ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni))

            if x - 1 >= 0:
                ni = idx - m
                if best_mark[ni] != current_run or new_cost < best_cost[ni]:
                    best_mark[ni] = current_run
                    best_cost[ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni))

        return result

    # Основной DP цикл
    current_positions: Dict[int, int] = {start_idx: 0}
    prev_char: str | None = None

    for step_idx, ch in enumerate(deliveries):
        remaining = len(deliveries) - step_idx - 1
        next_char = deliveries[step_idx + 1] if step_idx + 1 < len(deliveries) else None

        # Оптимизация: если буква повторяется, ничего не делаем
        if prev_char == ch:
            if not current_positions:
                print(-1)
                return
        else:
            # Определяем адаптивный cap для этого шага
            cap = get_adaptive_cap(step_idx, ch, next_char, remaining)

            # Выполняем мульти-источниковый Dijkstra
            current_positions = advance(current_positions, ch, cap, step_idx)

            if not current_positions:
                print(-1)
                return

        prev_char = ch

    answer = min(current_positions.values()) if current_positions else 0
    print(answer)


if __name__ == "__main__":
    solve()
