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
        
        # Сортируем точки: y (убывание), x (возрастание)
        points.sort(key=lambda p: (-p[1], p[0]))
        
        # Находим центроид дерева (корень для баланса)
        def find_centroid(root, parent, sizes):
            for child in adj[root]:
                if child != parent and sizes[child] > n // 2:
                    return find_centroid(child, root, sizes)
            return root
        
        def calc_sizes(v, parent):
            size = 1
            for u in adj[v]:
                if u != parent:
                    size += calc_sizes(u, v)
            sizes[v] = size
            return size
        
        sizes = [0] * n
        calc_sizes(0, -1)
        root = find_centroid(0, -1, sizes)
        
        # DFS с упорядочиванием детей по размеру поддерева
        visited = [False] * n
        order = []
        
        def dfs(v, parent):
            visited[v] = True
            order.append(v)
            
            # Получаем детей и их размеры
            children = []
            for u in adj[v]:
                if u != parent and not visited[u]:
                    children.append((sizes[u] if u in range(n) else 1, u))
            
            # Сортируем детей по размеру поддерева (большие первыми)
            children.sort(reverse=True)
            
            for _, child in children:
                dfs(child, v)
        
        # Пересчитываем размеры от нового корня
        sizes = [0] * n
        calc_sizes(root, -1)
        
        dfs(root, -1)
        
        # Сопоставляем
        result = [0] * n
        for i, v in enumerate(order):
            result[v] = points[i][2]
        
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
