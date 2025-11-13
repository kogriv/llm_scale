import math


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
            points.append((x, y))
        
        # Находим диаметр дерева (самый длинный путь)
        def bfs_farthest(start):
            from collections import deque
            q = deque([start])
            dist = [-1] * n
            dist[start] = 0
            farthest = start
            max_dist = 0
            
            while q:
                v = q.popleft()
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        q.append(u)
                        if dist[u] > max_dist:
                            max_dist = dist[u]
                            farthest = u
            
            return farthest, dist
        
        # Находим диаметр
        end1, _ = bfs_farthest(0)
        end2, dist_from_end1 = bfs_farthest(end1)
        
        # Восстанавливаем путь диаметра
        diameter_path = []
        current = end2
        parent = [-1] * n
        
        # BFS от end1 с сохранением родителей
        from collections import deque
        q = deque([end1])
        visited = [False] * n
        visited[end1] = True
        parent[end1] = -1
        
        while q:
            v = q.popleft()
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    parent[u] = v
                    q.append(u)
        
        # Восстанавливаем путь
        current = end2
        while current != -1:
            diameter_path.append(current)
            current = parent[current]
        
        diameter_path.reverse()
        
        # Находим выпуклую оболочку
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        def convex_hull():
            pts = sorted(enumerate(points), key=lambda p: (p[1][0], p[1][1]))
            if len(pts) <= 2:
                return [p[0] for p in pts]
            
            lower = []
            for i, p in pts:
                while len(lower) >= 2 and cross(points[lower[-2]], points[lower[-1]], p) <= 0:
                    lower.pop()
                lower.append(i)
            
            upper = []
            for i, p in reversed(pts):
                while len(upper) >= 2 and cross(points[upper[-2]], points[upper[-1]], p) <= 0:
                    upper.pop()
                upper.append(i)
            
            return lower[:-1] + upper[:-1]
        
        hull = convex_hull()
        
        # Размещаем вершины диаметра на оболочке
        assignment = [0] * n
        used_points = set()
        
        # Размещаем диаметр вдоль оболочки
        diameter_len = len(diameter_path)
        for i, v in enumerate(diameter_path):
            if i < len(hull):
                assignment[v] = hull[i]
                used_points.add(hull[i])
        
        # Размещаем остальные вершины
        remaining_vertices = [v for v in range(n) if v not in diameter_path]
        remaining_points = [p for p in range(n) if p not in used_points]
        
        for i, v in enumerate(remaining_vertices):
            if i < len(remaining_points):
                assignment[v] = remaining_points[i]
        
        # Вывод
        result = [assignment[i] + 1 for i in range(n)]
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
