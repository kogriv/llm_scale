import os
import subprocess
import time
from pathlib import Path

def run_test(test_file, solution_file, timeout=2):
    """Запускает решение на тесте"""
    try:
        start = time.time()
        result = subprocess.run(
            ['python', solution_file],
            stdin=open(test_file, 'r'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        elapsed = time.time() - start
        
        if result.returncode != 0:
            return False, f"Runtime Error: {result.stderr[:100]}", elapsed
        
        return True, result.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return False, "Time Limit Exceeded", timeout
    except Exception as e:
        return False, f"Error: {str(e)}", 0

def validate_output(output, n):
    """Проверяет корректность вывода"""
    try:
        parts = output.strip().split()
        if len(parts) != n:
            return False, f"Expected {n} numbers, got {len(parts)}"
        
        perm = [int(x) for x in parts]
        
        # Проверка что это перестановка 1..n
        if sorted(perm) != list(range(1, n + 1)):
            return False, "Not a valid permutation of 1..n"
        
        return True, "OK"
    except Exception as e:
        return False, f"Parse error: {str(e)}"

def check_planarity(edges, points, permutation):
    """Проверяет что вложение планарное (нет пересечений несмежных рёбер)"""
    
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def segments_intersect(p1, p2, p3, p4):
        """Проверяет пересечение отрезков (не включая концы)"""
        if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4:
            return False  # Общая вершина
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    # Сопоставляем вершинам точки
    assigned = [points[permutation[i] - 1] for i in range(len(permutation))]
    
    # Проверяем все пары рёбер
    for i in range(len(edges)):
        u1, v1 = edges[i]
        p1 = assigned[u1 - 1]
        p2 = assigned[v1 - 1]
        
        for j in range(i + 1, len(edges)):
            u2, v2 = edges[j]
            
            # Пропускаем смежные рёбра
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            
            p3 = assigned[u2 - 1]
            p4 = assigned[v2 - 1]
            
            if segments_intersect(p1, p2, p3, p4):
                return False, f"Edges ({u1},{v1}) and ({u2},{v2}) intersect"
    
    return True, "Planar"

def parse_test(test_file):
    """Читает тест из файла"""
    with open(test_file, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    t = int(lines[idx].strip())
    idx += 1
    
    n = int(lines[idx].strip())
    idx += 1
    
    edges = []
    for _ in range(n - 1):
        u, v = map(int, lines[idx].strip().split())
        edges.append((u, v))
        idx += 1
    
    points = []
    for _ in range(n):
        x, y = map(float, lines[idx].strip().split())
        points.append((x, y))
        idx += 1
    
    return n, edges, points

# Запуск тестов
import sys
solution_file = sys.argv[1] if len(sys.argv) > 1 else "solve_task4_yorder.py"
test_dir = Path("data/task4")

print(f"Проверка решения: {solution_file}")
print("=" * 80)

test_files = sorted(test_dir.glob("*.txt"))
passed = 0
failed = 0
total_time = 0

for test_file in test_files:
    test_name = test_file.stem
    
    # Читаем тест
    n, edges, points = parse_test(test_file)
    
    # Запускаем решение
    success, output, elapsed = run_test(test_file, solution_file)
    total_time += elapsed
    
    if not success:
        print(f"FAIL {test_name:30s} {output}")
        failed += 1
        continue
    
    # Проверяем формат вывода
    valid, msg = validate_output(output, n)
    if not valid:
        print(f"FAIL {test_name:30s} {msg}")
        failed += 1
        continue
    
    # Проверяем планарность
    perm = list(map(int, output.split()))
    planar, msg = check_planarity(edges, points, perm)
    
    if planar:
        print(f"OK {test_name:30s} n={n:4d} time={elapsed*1000:6.1f}ms")
        passed += 1
    else:
        print(f"FAIL {test_name:30s} {msg}")
        failed += 1

print("=" * 80)
print(f"Пройдено: {passed}/{len(test_files)}")
print(f"Провалено: {failed}/{len(test_files)}")
print(f"Общее время: {total_time:.2f}s")
print(f"Среднее время: {total_time/len(test_files)*1000:.1f}ms")
