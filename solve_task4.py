import sys
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
            points.append((x, y))
        
        # Простой подход: DFS обход дерева и размещение по выпуклой оболочке
        # Находим выпуклую оболочку точек
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        def convex_hull(points_with_idx):
            """Возвращает индексы точек выпуклой оболочки"""
            pts = sorted(points_with_idx, key=lambda p: (p[1][0], p[1][1]))
            if len(pts) <= 2:
                return [p[0] for p in pts]
            
            # Build lower hull
            lower = []
            for i, p in pts:
                while len(lower) >= 2 and cross(points[lower[-2]], points[lower[-1]], p) <= 0:
                    lower.pop()
                lower.append(i)
            
            # Build upper hull
            upper = []
            for i, p in reversed(pts):
                while len(upper) >= 2 and cross(points[upper[-2]], points[upper[-1]], p) <= 0:
                    upper.pop()
                upper.append(i)
            
            return lower[:-1] + upper[:-1]
        
        # Строим DFS обход от вершины 0
        visited = [False] * n
        dfs_order = []
        
        def dfs(v):
            visited[v] = True
            dfs_order.append(v)
            for u in adj[v]:
                if not visited[u]:
                    dfs(u)
        
        dfs(0)
        
        # Находим выпуклую оболочку
        hull_indices = convex_hull([(i, points[i]) for i in range(n)])
        
        # Если точек на оболочке меньше чем вершин дерева,
        # добавляем внутренние точки
        remaining = [i for i in range(n) if i not in hull_indices]
        
        # Размещаем вершины: сначала по оболочке, потом внутренние
        assignment = [0] * n
        
        if len(hull_indices) >= len(dfs_order):
            # Размещаем по порядку DFS на оболочку
            for i, v in enumerate(dfs_order):
                assignment[v] = hull_indices[i]
        else:
            # Размещаем часть на оболочку, часть внутрь
            for i, v in enumerate(dfs_order):
                if i < len(hull_indices):
                    assignment[v] = hull_indices[i]
                else:
                    assignment[v] = remaining[i - len(hull_indices)]
        
        # Вывод: для каждой вершины i выводим номер точки (1-based)
        result = [assignment[i] + 1 for i in range(n)]
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
