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
        
        # Просто сортируем точки лексикографически: (x, y)
        points.sort(key=lambda p: (p[0], p[1]))
        
        # DFS обход от вершины 0
        visited = [False] * n
        dfs_order = []
        
        def dfs(v):
            visited[v] = True
            dfs_order.append(v)
            # Сортируем детей для консистентности
            for u in sorted(adj[v]):
                if not visited[u]:
                    dfs(u)
        
        dfs(0)
        
        # Сопоставляем вершины в DFS порядке точкам в отсортированном порядке
        result = [0] * n
        for i, v in enumerate(dfs_order):
            result[v] = points[i][2]
        
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
