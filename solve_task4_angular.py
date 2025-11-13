import math
from collections import deque

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
        
        # Находим выпуклую оболочку точек
        def convex_hull(pts):
            """Возвращает индексы точек выпуклой оболочки против часовой стрелки"""
            if len(pts) <= 2:
                return list(range(len(pts)))
            
            points_with_idx = [(pts[i][0], pts[i][1], i) for i in range(len(pts))]
            points_with_idx.sort()
            
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            # Нижняя оболочка
            lower = []
            for p in points_with_idx:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            
            # Верхняя оболочка
            upper = []
            for p in reversed(points_with_idx):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            
            hull_points = lower[:-1] + upper[:-1]
            return [p[2] for p in hull_points]
        
        # Находим листья дерева
        def find_leaves():
            leaves = []
            for i in range(n):
                if len(adj[i]) == 1:
                    leaves.append(i)
            return leaves
        
        # DFS для подсчёта размеров поддеревьев
        def count_subtree_sizes(root):
            sizes = [0] * n
            visited = [False] * n
            
            def dfs(v, parent):
                visited[v] = True
                sizes[v] = 1
                for u in adj[v]:
                    if not visited[u]:
                        sizes[v] += dfs(u, v)
                return sizes[v]
            
            dfs(root, -1)
            return sizes
        
        # Находим центр дерева (вершину с минимальным максимальным поддеревом)
        def find_center():
            best = 0
            best_max_subtree = n
            
            for root in range(n):
                sizes = count_subtree_sizes(root)
                max_subtree = max(sizes[u] for u in adj[root]) if adj[root] else 0
                
                if max_subtree < best_max_subtree:
                    best_max_subtree = max_subtree
                    best = root
            
            return best
        
        # Стратегия: упорядочиваем точки по углу от центра масс
        center_x = sum(p[0] for p in points) / n
        center_y = sum(p[1] for p in points) / n
        
        points_with_angle = []
        for i, (x, y) in enumerate(points):
            angle = math.atan2(y - center_y, x - center_x)
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            points_with_angle.append((angle, dist, i + 1))
        
        # Сортируем по углу
        points_with_angle.sort()
        
        # DFS обход дерева от центра
        root = find_center()
        visited = [False] * n
        dfs_order = []
        
        def dfs(v):
            visited[v] = True
            dfs_order.append(v)
            for u in adj[v]:
                if not visited[u]:
                    dfs(u)
        
        dfs(root)
        
        # Сопоставляем вершины в порядке DFS точкам в порядке угла
        result = [0] * n
        for i, v in enumerate(dfs_order):
            result[v] = points_with_angle[i][2]
        
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
