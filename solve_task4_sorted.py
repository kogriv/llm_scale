def solve():
    t = int(input())
    
    for _ in range(t):
        n = int(input())
        
        if n == 1:
            print(1)
            continue
        
        # Читаем рёбра (но они не нужны для этой стратегии)
        for _ in range(n - 1):
            input()
        
        # Читаем точки
        points = []
        for i in range(n):
            x, y = map(float, input().split())
            points.append((x, y, i + 1))  # (x, y, исходный_индекс)
        
        # Сортируем точки по y-координате (по возрастанию)
        sorted_points = sorted(enumerate(points, 1), key=lambda p: p[1][1])
        
        # Создаём перестановку: вершина i получает точку с i-м порядковым номером по y
        # sorted_points[i-1][0] - это исходный номер точки с i-м наименьшим y
        result = [sorted_points[i][1][2] for i in range(n)]
        
        print(' '.join(map(str, result)))


if __name__ == "__main__":
    solve()
