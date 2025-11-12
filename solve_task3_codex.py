import sys
import heapq
from typing import Dict, List


def solve() -> None:
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

    def advance(front: Dict[int, int], target_char: str) -> Dict[int, int]:
        nonlocal mark_id
        nonlocal dist_run
        targets = addresses[target_char]
        if not targets:
            return {}

        mark_id += 1
        current_mark = mark_id
        for idx in targets:
            target_mark[idx] = current_mark

        remaining = len(targets)
        result: Dict[int, int] = {}

        # Стоимости посещения клеток во время Dijkstra
        dist_run += 1
        current_run = dist_run
        heap: List[tuple[int, int]] = []
        for idx, cost in front.items():
            if best_mark[idx] != current_run or cost < best_cost[idx]:
                best_mark[idx] = current_run
                best_cost[idx] = cost
                heapq.heappush(heap, (cost, idx))

        while heap and remaining:
            cost, idx = heapq.heappop(heap)
            if best_mark[idx] != current_run or cost != best_cost[idx]:
                continue

            if target_mark[idx] == current_mark and idx not in result:
                result[idx] = cost
                remaining -= 1
                if remaining == 0:
                    break

            x, y = idx_to_xy(idx)
            if y + 1 < m:
                ni = idx + 1
                new_cost = cost + 1
                if best_mark[ni] != current_run or new_cost < best_cost[ni]:
                    best_mark[ni] = current_run
                    best_cost[ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni))
            if y - 1 >= 0:
                ni = idx - 1
                new_cost = cost + 1
                if best_mark[ni] != current_run or new_cost < best_cost[ni]:
                    best_mark[ni] = current_run
                    best_cost[ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni))
            if x + 1 < n:
                ni = idx + m
                new_cost = cost + 1
                if best_mark[ni] != current_run or new_cost < best_cost[ni]:
                    best_mark[ni] = current_run
                    best_cost[ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni))
            if x - 1 >= 0:
                ni = idx - m
                new_cost = cost + 1
                if best_mark[ni] != current_run or new_cost < best_cost[ni]:
                    best_mark[ni] = current_run
                    best_cost[ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni))

        return result

    current_positions: Dict[int, int] = {start_idx: 0}

    prev_char: str | None = None

    for ch in deliveries:
        if prev_char == ch:
            # Повторяющиеся буквы не уменьшают стоимости: уже посчитаны
            if not current_positions:
                print(-1)
                return
        else:
            current_positions = advance(current_positions, ch)
            if not current_positions:
                print(-1)
                return
        prev_char = ch

    answer = min(current_positions.values()) if current_positions else 0
    print(answer)


if __name__ == "__main__":
    solve()
