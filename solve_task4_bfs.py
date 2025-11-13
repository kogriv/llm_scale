from collections import deque

def solve():
    t = int(input())
    
    for _ in range(t):
        n = int(input())
        
        if n == 1:
            print(1)
            continue
        
        # Читаем рёбра
        adj = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = map(int, input().split())
            u -= 1
            v -= 1
            adj[u].append(v)
            adj[v].append(u)
        
        # Читаем точки
        points = []
        for i in range(n):
            x, y = map(float, input().split())
            points.append((x, y, i + 1))
        
        # Сортируем точки: сначала по y (убывание), затем по x (возрастание)
        points.sort(key=lambda p: (-p[1], p[0]))
        
        # Находим корень дерева - вершину с наименьшей степенью, 
        # или первый лист, или просто вершину 0
        def find_root():
            # Ищем лист
            for i in range(n):
                if len(adj[i]) == 1:
                    return i
            # Если нет листьев (невозможно), берём 0
            return 0
        
        root = find_root()
        
        # BFS обход для упорядочивания вершин по слоям
        visited = [False] * n
        queue = deque([root])
        visited[root] = True
        bfs_order = []
        
        while queue:
            v = queue.popleft()
            bfs_order.append(v)
            
            # Сортируем детей для консистентности
            children = sorted([u for u in adj[v] if not visited[u]])
            
            for u in children:
                visited[u] = True
                queue.append(u)
        
        # Сопоставляем вершины в BFS порядке точкам в отсортированном порядке
        result = [0] * n
        for i, v in enumerate(bfs_order):
            result[v] = points[i][2]
        
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
