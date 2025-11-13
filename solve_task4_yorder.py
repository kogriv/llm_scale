def solve():
    t = int(input())
    
    for _ in range(t):
        n = int(input())
        
        if n == 1:
            print(1)
            continue
        
        # Читаем рёбра (не используем в этой стратегии)
        for _ in range(n - 1):
            input()
        
        # Читаем точки и запоминаем их исходные номера
        points_with_idx = []
        for i in range(n):
            x, y = map(float, input().split())
            points_with_idx.append((y, x, i + 1))  # (y, x, номер_точки)
        
        # Сортируем по y (убывание), затем по x (возрастание)
        points_with_idx.sort(key=lambda p: (-p[0], p[1]))
        
        # Вершина i получает точку с соответствующим порядком
        result = [points_with_idx[i][2] for i in range(n)]
        
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
