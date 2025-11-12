from collections import deque, defaultdict
import heapq
import sys

# Параметры: можно передать max_targets и max_states через аргументы
# Пример: python solve_task3_alt.py 100 40
MAX_TARGETS_DEFAULT = 100

# Разбор аргументов
try:
    MAX_TARGETS_PER_BFS = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_TARGETS_DEFAULT
except Exception:
    MAX_TARGETS_PER_BFS = MAX_TARGETS_DEFAULT


def solve(MAX_TARGETS_PER_BFS: int = MAX_TARGETS_DEFAULT):
    # Чтение входных данных
    n, m = map(int, input().split())
    sx, sy = map(int, input().split())

    grid = [input().strip() for _ in range(n)]
    s = input().strip()

    def to_idx(x, y):
        return (x - 1) * m + (y - 1)

    def idx_xy(idx):
        return divmod(idx, m)

    addresses_idx = defaultdict(list)
    for i in range(n):
        row = grid[i]
        base = i * m
        for j in range(m):
            addresses_idx[row[j]].append(base + j)

    visited_mark = [[0] * m for _ in range(n)]
    run_id = [0]

    def bfs_to_targets_idx(start_idx, targets_list):
        if not targets_list:
            return {}
        tx_set = set(targets_list)
        found = {}

        run_id[0] += 1
        mark = run_id[0]

        sx0, sy0 = idx_xy(start_idx)
        dq = deque([start_idx, None])
        visited_mark[sx0][sy0] = mark

        if start_idx in tx_set:
            found[start_idx] = 0
            tx_set.remove(start_idx)
            if not tx_set:
                return found

        dist = 0
        cap = min(len(targets_list), MAX_TARGETS_PER_BFS)
        while dq and tx_set and len(found) < cap:
            cur = dq.popleft()
            if cur is None:
                dist += 1
                if dq:
                    dq.append(None)
                continue

            x, y = idx_xy(cur)
            # 4 соседних клетки
            if y + 1 < m and visited_mark[x][y + 1] != mark:
                ni = cur + 1
                visited_mark[x][y + 1] = mark
                if ni in tx_set:
                    found[ni] = dist + 1
                    tx_set.remove(ni)
                dq.append(ni)
            if y - 1 >= 0 and visited_mark[x][y - 1] != mark:
                ni = cur - 1
                visited_mark[x][y - 1] = mark
                if ni in tx_set:
                    found[ni] = dist + 1
                    tx_set.remove(ni)
                dq.append(ni)
            if x + 1 < n and visited_mark[x + 1][y] != mark:
                ni = cur + m
                visited_mark[x + 1][y] = mark
                if ni in tx_set:
                    found[ni] = dist + 1
                    tx_set.remove(ni)
                dq.append(ni)
            if x - 1 >= 0 and visited_mark[x - 1][y] != mark:
                ni = cur - m
                visited_mark[x - 1][y] = mark
                if ni in tx_set:
                    found[ni] = dist + 1
                    tx_set.remove(ni)
                dq.append(ni)

        return found

    start_idx = to_idx(sx, sy)
    current_positions = {start_idx: 0}

    grid_size = n * m
    seq_len = len(s)
    # MAX_STATES подберём умеренно-консервативно
    if seq_len > 150:
        MAX_STATES = 12
    elif grid_size > 30000:
        MAX_STATES = 15
    elif grid_size > 20000 or seq_len > 100:
        MAX_STATES = 18
    elif grid_size > 5000 or seq_len > 50:
        MAX_STATES = 30
    else:
        MAX_STATES = 50

    for delivery_type in s:
        next_positions = {}
        targets = addresses_idx[delivery_type]

        for cur_idx, cost_so_far in current_positions.items():
            dists = bfs_to_targets_idx(cur_idx, targets)
            for t, dist in dists.items():
                new_cost = cost_so_far + dist
                if t not in next_positions or new_cost < next_positions[t]:
                    next_positions[t] = new_cost

        if len(next_positions) > MAX_STATES:
            current_positions = dict(heapq.nsmallest(MAX_STATES, next_positions.items(), key=lambda x: x[1]))
        else:
            current_positions = next_positions

    print(min(current_positions.values()))


if __name__ == "__main__":
    solve(MAX_TARGETS_PER_BFS)
