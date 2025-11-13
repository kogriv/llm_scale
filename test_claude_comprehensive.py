"""Comprehensive тесты для Claude решения (без проблем с кодировкой)"""
import subprocess
import time

def run_test(test_content, solver="solve_task3_claude.py", timeout=10):
    try:
        start = time.time()
        result = subprocess.run(
            ['python', solver],
            input=test_content,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            return int(result.stdout.strip()), elapsed
        return None, elapsed
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout
    except Exception as e:
        return f"ERROR: {e}", 0

def test1_abundant_type():
    """Обильный тип 'a' - 500+ клеток"""
    n, m = 50, 50
    grid = ['a' * m for _ in range(n)]
    s = 'a' * 10
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test2_transitions():
    """Частые переходы между типами"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        if i < 10:
            grid.append('a' * m)
        elif i < 20:
            grid.append('b' * m)
        else:
            grid.append('c' * m)
    s = 'abc' * 10
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test3_threshold_490():
    """Ровно 490 клеток (ниже порога 500)"""
    n, m = 49, 10
    grid = ['a' * m for _ in range(n)]
    s = 'a' * 8
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test4_threshold_510():
    """Ровно 510 клеток (выше порога 500)"""
    n, m = 51, 10
    grid = ['a' * m for _ in range(n)]
    s = 'a' * 8
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test5_clusters():
    """Кластеризация типов"""
    n, m = 40, 40
    grid = []
    for i in range(n):
        row = ""
        for j in range(m):
            if i < 20 and j < 20:
                row += 'a'
            elif i < 20:
                row += 'b'
            elif j < 20:
                row += 'c'
            else:
                row += 'd'
        grid.append(row)
    s = 'abcd' * 5
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test6_checkerboard():
    """Шахматный паттерн"""
    n, m = 20, 20
    grid = []
    for i in range(n):
        row = ""
        for j in range(m):
            row += 'a' if (i + j) % 2 == 0 else 'b'
        grid.append(row)
    s = 'ab' * 10
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test7_max_grid():
    """Большая карта 200x200"""
    n, m = 200, 200
    grid = []
    for i in range(n):
        row = ""
        for j in range(m):
            row += chr(ord('a') + ((i + j) % 26))
        grid.append(row)
    s = 'abc' * 5
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test8_repeating_pattern():
    """Повторяющийся паттерн"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        row = ""
        for j in range(m):
            row += chr(ord('a') + ((i * m + j) % 3))
        grid.append(row)
    s = 'abc' * 10
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test9_long_sequence():
    """Длинная последовательность"""
    n, m = 50, 50
    grid = []
    for i in range(n):
        row = ""
        for j in range(m):
            row += chr(ord('a') + ((i + j) % 10))
        grid.append(row)
    s = 'abcdefghij' * 15
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test10_sparse():
    """Редкие типы"""
    n, m = 50, 50
    grid = []
    for i in range(n):
        row = 'z' * m
        grid.append(row)
    # Добавим несколько клеток 'a'
    grid[0] = 'a' + grid[0][1:]
    grid[24] = grid[24][:24] + 'a' + grid[24][25:]
    grid[49] = grid[49][:49] + 'a'
    s = 'aaa'
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

print("=" * 80)
print("COMPREHENSIVE TESTS - Claude Solution")
print("=" * 80)

tests = [
    ("Abundant type (500+ cells)", test1_abundant_type()),
    ("Frequent transitions", test2_transitions()),
    ("Threshold boundary 490", test3_threshold_490()),
    ("Threshold boundary 510", test4_threshold_510()),
    ("Clustered types", test5_clusters()),
    ("Checkerboard pattern", test6_checkerboard()),
    ("Large grid 200x200", test7_max_grid()),
    ("Repeating pattern abc*10", test8_repeating_pattern()),
    ("Long sequence 150 chars", test9_long_sequence()),
    ("Sparse targets", test10_sparse()),
]

results = []
for name, test_content in tests:
    print(f"\n{name}")
    result, elapsed = run_test(test_content)
    if result == "TIMEOUT":
        print(f"  TIMEOUT (>{elapsed:.1f}s)")
        results.append((name, None, "TIMEOUT"))
    elif isinstance(result, str):
        print(f"  ERROR: {result}")
        results.append((name, None, "ERROR"))
    else:
        print(f"  OK: {result} ({elapsed:.3f}s)")
        results.append((name, result, elapsed))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
ok_count = sum(1 for _, r, _ in results if isinstance(r, int))
timeout_count = sum(1 for _, _, s in results if s == "TIMEOUT")
error_count = sum(1 for _, _, s in results if s == "ERROR")
print(f"Total tests: {len(tests)}")
print(f"OK:          {ok_count}")
print(f"TIMEOUT:     {timeout_count}")
print(f"ERROR:       {error_count}")
if ok_count > 0:
    avg_time = sum(e for _, r, e in results if isinstance(r, int) and isinstance(e, float)) / ok_count
    print(f"Avg time:    {avg_time:.3f}s")
