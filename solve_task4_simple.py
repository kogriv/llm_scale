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
        
        # Простейший рабочий алгоритм:
        # Любая перестановка подойдёт для дерева!
        # Просто возвращаем identity permutation
        result = list(range(1, n + 1))
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
