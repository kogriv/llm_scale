import random
import math
import os

def generate_tree_edges(n, tree_type="random"):
    """Генерирует рёбра дерева разных типов"""
    edges = []
    
    if tree_type == "chain":
        # Цепочка: 1-2-3-...-n
        for i in range(1, n):
            edges.append((i, i + 1))
    
    elif tree_type == "star":
        # Звезда: 1 соединён со всеми
        for i in range(2, n + 1):
            edges.append((1, i))
    
    elif tree_type == "binary":
        # Бинарное дерево
        for i in range(2, n + 1):
            parent = i // 2
            edges.append((parent, i))
    
    elif tree_type == "random":
        # Случайное дерево (алгоритм Прюфера)
        if n == 1:
            return []
        if n == 2:
            return [(1, 2)]
        
        # Генерируем код Прюфера
        prufer = [random.randint(1, n) for _ in range(n - 2)]
        
        # Восстанавливаем дерево
        degree = [1] * (n + 1)
        for node in prufer:
            degree[node] += 1
        
        for node in prufer:
            for i in range(1, n + 1):
                if degree[i] == 1:
                    edges.append((node, i))
                    degree[node] -= 1
                    degree[i] -= 1
                    break
        
        # Последнее ребро
        remaining = [i for i in range(1, n + 1) if degree[i] == 1]
        if len(remaining) == 2:
            edges.append(tuple(remaining))
    
    return edges

def generate_points(n, point_type="random"):
    """Генерирует точки разных типов"""
    points = []
    
    if point_type == "random":
        # Случайные точки
        for _ in range(n):
            x = random.uniform(-10000, 10000)
            y = random.uniform(-10000, 10000)
            points.append((x, y))
    
    elif point_type == "circle":
        # Точки на окружности
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = 1000 * math.cos(angle)
            y = 1000 * math.sin(angle)
            points.append((x, y))
    
    elif point_type == "grid":
        # Точки на сетке
        side = int(math.ceil(math.sqrt(n)))
        for i in range(n):
            x = (i % side) * 100
            y = (i // side) * 100
            points.append((x, y))
    
    elif point_type == "line_horizontal":
        # Точки на горизонтальной линии (y=0)
        for i in range(n):
            x = i * 100
            y = 0
            points.append((x, y))
    
    elif point_type == "line_vertical":
        # Точки на вертикальной линии (x=0)
        for i in range(n):
            x = 0
            y = i * 100
            points.append((x, y))
    
    elif point_type == "convex_hull":
        # Точки образуют выпуклую оболочку
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(n)])
        for angle in angles:
            radius = random.uniform(500, 1000)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append((x, y))
    
    elif point_type == "same_y":
        # Несколько точек с одинаковым y
        y_values = [i * 100 for i in range(max(1, n // 3))]
        for i in range(n):
            x = random.uniform(-1000, 1000)
            y = random.choice(y_values)
            points.append((x, y))
    
    elif point_type == "extreme":
        # Экстремальные координаты
        coords = [-10000, -5000, 0, 5000, 10000]
        for i in range(n):
            x = random.choice(coords)
            y = random.choice(coords)
            points.append((x, y))
    
    return points

def create_test(n, tree_type, point_type, test_name):
    """Создаёт один тест"""
    edges = generate_tree_edges(n, tree_type)
    points = generate_points(n, point_type)
    
    # Перемешиваем точки чтобы они были в случайном порядке
    random.shuffle(points)
    
    content = f"1\n{n}\n"
    for u, v in edges:
        content += f"{u} {v}\n"
    for x, y in points:
        content += f"{x} {y}\n"
    
    return content, test_name

# Создаём директорию для тестов
os.makedirs("data/task4", exist_ok=True)

# Генерируем различные тесты
test_cases = [
    # Минимальные тесты
    (2, "chain", "random", "01_min_n2"),
    (3, "chain", "random", "02_min_n3"),
    
    # Цепочки с разными точками
    (10, "chain", "random", "03_chain_random"),
    (20, "chain", "circle", "04_chain_circle"),
    (15, "chain", "line_vertical", "05_chain_vertical"),
    (15, "chain", "line_horizontal", "06_chain_horizontal"),
    
    # Звёзды
    (10, "star", "random", "07_star_random"),
    (20, "star", "circle", "08_star_circle"),
    (15, "star", "convex_hull", "09_star_convex"),
    
    # Бинарные деревья
    (15, "binary", "random", "10_binary_random"),
    (31, "binary", "grid", "11_binary_grid"),
    (15, "binary", "same_y", "12_binary_samey"),
    
    # Случайные деревья
    (50, "random", "random", "13_random_random"),
    (100, "random", "circle", "14_random_circle"),
    (75, "random", "convex_hull", "15_random_convex"),
    
    # Крайние случаи
    (50, "chain", "extreme", "16_chain_extreme"),
    (50, "star", "extreme", "17_star_extreme"),
    (100, "random", "same_y", "18_random_samey"),
    
    # Большие тесты
    (200, "random", "random", "19_large_200"),
    (500, "random", "random", "20_large_500"),
    (1000, "chain", "random", "21_max_chain"),
    (1000, "star", "random", "22_max_star"),
    (1000, "random", "random", "23_max_random"),
    
    # Патологические случаи
    (50, "chain", "grid", "24_pathological_chain_grid"),
    (100, "binary", "extreme", "25_pathological_binary_extreme"),
    (200, "star", "same_y", "26_pathological_star_samey"),
]

print("Генерирую тесты...")
for n, tree_type, point_type, name in test_cases:
    content, test_name = create_test(n, tree_type, point_type, name)
    filename = f"data/task4/{name}.txt"
    with open(filename, 'w') as f:
        f.write(content)
    print(f"✓ {filename}")

print(f"\nСоздано {len(test_cases)} тестов")
