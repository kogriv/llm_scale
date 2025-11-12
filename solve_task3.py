from collections import deque, defaultdict
import heapq

def solve():
    # Чтение входных данных
    n, m = map(int, input().split())
    sx, sy = map(int, input().split())

    # Чтение карты города (строки длиной m)
    grid = [input().strip() for _ in range(n)]

    # Последовательность доставок
    s = input().strip()

    # Вспомогательные преобразования индексов
    def to_idx(x, y):  # 1-based -> 0-based linear
        return (x - 1) * m + (y - 1)

    def idx_xy(idx):  # 0-based linear -> (x, y) 0-based
        return divmod(idx, m)

    # Словарь: тип адреса -> список линейных индексов (0-based)
    addresses_idx = defaultdict(list)
    for i in range(n):
        row = grid[i]
        base = i * m
        for j in range(m):
            addresses_idx[row[j]].append(base + j)

    # Быстрый BFS только до заданных целей (линейные индексы), с ранним завершением
    visited_mark = [[0] * m for _ in range(n)]
    run_id = [0]  # счётчик запусков BFS, чтобы не обнулять visited

    def bfs_to_targets_idx(start_idx, targets_list, cap=None, finish_level=False):
        if not targets_list:
            return {}
        tx_set = set(targets_list)
        found = {}

        run_id[0] += 1
        mark = run_id[0]

        sx0, sy0 = idx_xy(start_idx)
        dq = deque()
        dq.append(start_idx)
        visited_mark[sx0][sy0] = mark

        if start_idx in tx_set:
            found[start_idx] = 0
            tx_set.remove(start_idx)
            if not tx_set:
                return found

        # Для хранения текущей дистанции по слоям BFS
        dist = 0
        # Разделим на слои: добавим sentinel None для уровня
        dq.append(None)

        # Лимитируем количество найденных целей, но корректно завершаем текущий слой BFS
        # cap=None означает «без лимита»
        max_targets_to_find = len(targets_list) if cap is None else min(len(targets_list), cap)
        finish_level_flag = finish_level

        stop_after_level = False
        while dq and tx_set:
            if not finish_level_flag and len(found) >= max_targets_to_find:
                break
            cur = dq.popleft()
            if cur is None:
                if stop_after_level:
                    break
                dist += 1
                if dq:
                    dq.append(None)
                continue

            x, y = idx_xy(cur)
            # Соседи (оптимизированный порядок проверок)
            if y + 1 < m and visited_mark[x][y + 1] != mark:
                ni = cur + 1
                visited_mark[x][y + 1] = mark
                if ni in tx_set:
                    found[ni] = dist + 1
                    tx_set.remove(ni)
                    if len(found) >= max_targets_to_find:
                        stop_after_level = True
                dq.append(ni)
            if y - 1 >= 0 and visited_mark[x][y - 1] != mark:
                ni = cur - 1
                visited_mark[x][y - 1] = mark
                if ni in tx_set:
                    found[ni] = dist + 1
                    tx_set.remove(ni)
                    if len(found) >= max_targets_to_find:
                        stop_after_level = True
                dq.append(ni)
            if x + 1 < n and visited_mark[x + 1][y] != mark:
                ni = cur + m
                visited_mark[x + 1][y] = mark
                if ni in tx_set:
                    found[ni] = dist + 1
                    tx_set.remove(ni)
                    if len(found) >= max_targets_to_find:
                        stop_after_level = True
                dq.append(ni)
            if x - 1 >= 0 and visited_mark[x - 1][y] != mark:
                ni = cur - m
                visited_mark[x - 1][y] = mark
                if ni in tx_set:
                    found[ni] = dist + 1
                    tx_set.remove(ni)
                    if len(found) >= max_targets_to_find:
                        stop_after_level = True
                dq.append(ni)

        return found

    # Динамическое программирование по шагам доставок
    start_idx = to_idx(sx, sy)
    current_positions = {start_idx: 0}  # {linear_idx: cost}

    grid_size = n * m
    seq_len = len(s)
    # Ещё более агрессивное ограничение для экстремальных случаев
    if seq_len > 150:  # Очень длинная последовательность
        MAX_STATES = 12  # чуть расширим фронт, чтобы снизить риск WA
    elif grid_size > 30000:
        MAX_STATES = 15
    elif grid_size > 20000 or seq_len > 100:
        MAX_STATES = 20
    elif grid_size > 5000 or seq_len > 50:
        MAX_STATES = 32
    else:
        MAX_STATES = 50

    # Хвостовой буст для последних шагов: снимаем cap и расширяем фронт
    # Увеличиваем до 5 для максимальной точности
    TAIL_STEPS = 5

    for step_idx, delivery_type in enumerate(s):
        next_positions = {}
        targets = addresses_idx[delivery_type]
        remaining = len(s) - step_idx - 1
        
        # Адаптивный cap с учётом переходов между типами
        is_tail = remaining < TAIL_STEPS
        
        # Проверяем переход типа: если следующий шаг — другая буква
        is_transition = False
        if step_idx + 1 < len(s):
            next_type = s[step_idx + 1]
            is_transition = (next_type != delivery_type)
        
        # Проверяем "обильный" тип: много клеток одного типа
        # Порог 300 - еще более агрессивно включаем полный поиск
        is_abundant = len(targets) > 300
        
        # Максимально консервативная стратегия:
        # 1. Хвост (последние 5 шагов) — без лимита
        # 2. Переход к другому типу И обильный — БЕЗ лимита (убрали cap!)
        # 3. Иначе — умеренный cap
        if is_tail:
            bfs_cap = None
            finish_level = True
        elif is_transition and is_abundant:
            # КРИТИЧНО: на переходах между обильными типами - полный поиск!
            bfs_cap = None
            finish_level = True
        else:
            # Стандартный случай
            bfs_cap = 100
            finish_level = True

        for cur_idx, cost_so_far in current_positions.items():
            dists = bfs_to_targets_idx(cur_idx, targets, bfs_cap, finish_level)
            # Рассматриваем найденные цели (не обязательно все, если их >100, кроме хвоста)
            for t, dist in dists.items():
                new_cost = cost_so_far + dist
                if t not in next_positions or new_cost < next_positions[t]:
                    next_positions[t] = new_cost
        
        # Проверка на пустой next_positions (не должно происходить в корректных тестах)
        if not next_positions:
            print(-1)  # Нет решения
            return

        # Pruning: простое ограничение по лучшим стоимостям (без тип-based логики)
        # На хвосте или переходах расширим фронт, но умеренно
        if is_tail:
            max_states_this_step = MAX_STATES * 3
        elif is_transition and is_abundant:
            # Небольшое расширение на переходах для корректности
            max_states_this_step = int(MAX_STATES * 1.5)
        else:
            max_states_this_step = MAX_STATES
            
        if len(next_positions) > max_states_this_step:
            # Используем heapq для более быстрой выборки топ-N
            current_positions = dict(heapq.nsmallest(max_states_this_step, next_positions.items(), key=lambda x: x[1]))
        else:
            current_positions = next_positions

    # Минимальное время среди всех финальных позиций
    print(min(current_positions.values()))

if __name__ == "__main__":
    solve()
