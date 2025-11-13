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
        
        # Проверка пересечения отрезков
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def segments_intersect(A, B, C, D):
            """Проверяет пересекаются ли отрезки AB и CD (не включая концы)"""
            if A == C or A == D or B == C or B == D:
                return False  # Общая вершина
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        def check_valid(assignment):
            """Проверяет что текущее размещение не создаёт пересечений"""
            # Проверяем все пары рёбер
            for i in range(len(edges)):
                u1, v1 = edges[i]
                if assignment[u1] == -1 or assignment[v1] == -1:
                    continue
                p1 = points[assignment[u1]]
                p2 = points[assignment[v1]]
                
                for j in range(i + 1, len(edges)):
                    u2, v2 = edges[j]
                    if assignment[u2] == -1 or assignment[v2] == -1:
                        continue
                    
                    # Пропускаем смежные рёбра
                    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                        continue
                    
                    p3 = points[assignment[u2]]
                    p4 = points[assignment[v2]]
                    
                    if segments_intersect(p1, p2, p3, p4):
                        return False
            return True
        
        # Жадный поиск с backtracking
        assignment = [-1] * n
        used = [False] * n
        
        def backtrack(v_idx):
            if v_idx == n:
                return check_valid(assignment)
            
            for p_idx in range(n):
                if not used[p_idx]:
                    assignment[v_idx] = p_idx
                    used[p_idx] = True
                    
                    if check_valid(assignment):
                        if backtrack(v_idx + 1):
                            return True
                    
                    assignment[v_idx] = -1
                    used[p_idx] = False
            
            return False
        
        backtrack(0)
        
        # Вывод
        result = [assignment[i] + 1 for i in range(n)]
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
