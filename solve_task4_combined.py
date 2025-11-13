from collections import deque
import math

def solve():
    t = int(input())
    
    for _ in range(t):
        n = int(input())
        
        if n == 1:
            print(1)
            continue
        
        # Читаем рёбра
        edges = []
        adj = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = map(int, input().split())
            u -= 1
            v -= 1
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
        
        # Читаем точки
        points = []
        for i in range(n):
            x, y = map(float, input().split())
            points.append((x, y, i + 1))
        
        # Определяем тип дерева
        degrees = [len(adj[i]) for i in range(n)]
        max_degree = max(degrees)
        leaves = sum(1 for d in degrees if d == 1)
        
        # Сортируем точки
        # Основная стратегия: y (убывание), затем x (возрастание)
        points.sort(key=lambda p: (-p[1], p[0]))
        
        # Цепочка: максимальная степень 2
        if max_degree <= 2:
            # Начинаем с листа
            root = degrees.index(1)
            
            # Обходим цепочку
            visited = [False] * n
            chain = []
            curr = root
            visited[curr] = True
            chain.append(curr)
            
            while len(chain) < n:
                for next_v in adj[curr]:
                    if not visited[next_v]:
                        visited[next_v] = True
                        chain.append(next_v)
                        curr = next_v
                        break
            
            result = [0] * n
            for i, v in enumerate(chain):
                result[v] = points[i][2]
        
        # Звезда: одна вершина соединена со всеми
        elif max_degree == n - 1:
            center = degrees.index(max_degree)
            
            # Центр получает среднюю точку
            mid = n // 2
            result = [0] * n
            result[center] = points[mid][2]
            
            # Листья получают остальные точки
            leaf_idx = 0
            for v in range(n):
                if v != center:
                    if leaf_idx == mid:
                        leaf_idx += 1
                    result[v] = points[leaf_idx][2]
                    leaf_idx += 1
        
        # Общий случай: BFS обход
        else:
            # Начинаем с вершины наименьшей степени
            root = degrees.index(min(degrees))
            
            # BFS обход
            visited = [False] * n
            queue = deque([root])
            visited[root] = True
            bfs_order = []
            
            while queue:
                v = queue.popleft()
                bfs_order.append(v)
                
                # Сортируем соседей по углу относительно текущей позиции
                # Чтобы они располагались "по кругу"
                if len(bfs_order) > 1 and v != root:
                    # Упорядочиваем детей случайным образом для разнообразия
                    children = sorted([u for u in adj[v] if not visited[u]])
                else:
                    children = sorted([u for u in adj[v] if not visited[u]])
                
                for u in children:
                    visited[u] = True
                    queue.append(u)
            
            result = [0] * n
            for i, v in enumerate(bfs_order):
                result[v] = points[i][2]
        
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
