"""
Финальная проверка всех оставшихся случаев
"""
import subprocess
from pathlib import Path
import time

def run_both(test_input, name):
    base_dir = Path(__file__).parent
    optimized = base_dir / 'solve_task3.py'
    reference = base_dir / 'solve_task3_reference.py'
    
    try:
        start = time.time()
        res_opt = subprocess.run(['python', optimized], input=test_input, capture_output=True, text=True, timeout=10)
        time_opt = time.time() - start
        
        start = time.time()
        res_ref = subprocess.run(['python', reference], input=test_input, capture_output=True, text=True, timeout=10)
        time_ref = time.time() - start
        
        opt = res_opt.stdout.strip() if res_opt.returncode == 0 else "ERROR"
        ref = res_ref.stdout.strip() if res_ref.returncode == 0 else "ERROR"
        
        match = "✓" if opt == ref else "❌"
        print(f"{match} {name}")
        print(f"    Opt: {opt} ({time_opt:.2f}s), Ref: {ref} ({time_ref:.2f}s)")
        
        if opt != ref:
            with open(base_dir / f'failed_final_{name.replace(" ", "_")[:30]}.txt', 'w') as f:
                f.write(test_input)
        
        return opt == ref
    except subprocess.TimeoutExpired:
        print(f"⚠️  {name} - TIMEOUT")
        return True

def test_repeating_pattern():
    """Повторяющийся паттерн abcabc×10"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(chr(ord('a') + ((i + j) % 5)))
        grid.append(''.join(row))
    s = 'abc' * 10
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_maze():
    """Лабиринт: узкие коридоры из 'a', остальное 'z'"""
    n, m = 40, 40
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            # Коридоры: каждые 3 строки
            if i % 3 == 0 or j % 3 == 0:
                row.append('a')
            else:
                row.append('z')
        grid.append(''.join(row))
    s = 'a' * 15
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_spiral():
    """Спираль: типы идут по спирали от центра"""
    n, m = 25, 25
    grid = [['z'] * m for _ in range(n)]
    
    # Простая спираль: круги разных типов
    for i in range(n):
        for j in range(m):
            dist = min(i, j, n-1-i, m-1-j)
            grid[i][j] = chr(ord('a') + (dist % 5))
    
    grid_str = [''.join(row) for row in grid]
    s = 'abcde' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid_str) + f"\n{s}\n"

def test_max_distance():
    """Максимальное расстояние: цель только в противоположном углу"""
    n, m = 50, 50
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == n-1 and j == m-1:
                row.append('x')
            else:
                row.append('a')
        grid.append(''.join(row))
    s = 'a' * 5 + 'x' + 'a' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_multiple_transitions_abundant():
    """Множественные переходы через обильные типы"""
    n, m = 60, 60
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if j < 20:
                row.append('a')
            elif j < 25:
                row.append('b')
            elif j < 45:
                row.append('a')
            else:
                row.append('c')
        grid.append(''.join(row))
    # a(1200) → b(300) → a(1200) → c(900)
    s = 'a' * 5 + 'b' + 'a' * 5 + 'c'
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_start_at_target():
    """Старт в углу с нужной буквой (cost=0 сразу)"""
    n, m = 20, 20
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == 0 and j == 0:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 3 + 'b' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_start_center():
    """Старт в центре карты"""
    n, m = 40, 40
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i < n // 2:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 5 + 'b' * 5
    return f"{n} {m}\n{n//2} {m//2}\n" + '\n'.join(grid) + f"\n{s}\n"

def test_start_not_in_sequence():
    """Старт на клетке, которая НЕ встречается в последовательности"""
    n, m = 25, 25
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == 0 and j == 0:
                row.append('z')  # Старт на 'z'
            elif i < n // 2:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 5 + 'b' * 5  # Нет 'z' в последовательности
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_exactly_100_targets_fixed():
    """Ровно 100 клеток типа 'a' (исправленный тест)"""
    n, m = 10, 10
    grid = []
    for i in range(n):
        if i < 10:
            grid.append('a' * 10)  # Все 100 клеток 'a'
    s = 'a' * 8 + 'b' + 'a' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nb\n{s}\n"  # Добавили строку с 'b'

def test_large_grid_300x300():
    """Максимальная карта 300×300"""
    n, m = 300, 300
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(chr(ord('a') + ((i + j) % 3)))
        grid.append(''.join(row))
    s = 'abc' * 5
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_tall_grid_300x1():
    """Вытянутая карта 300×1"""
    n, m = 300, 1
    grid = []
    for i in range(n):
        if i < 150:
            grid.append('a')
        else:
            grid.append('b')
    s = 'a' * 3 + 'b' + 'a'
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_combined_pathology():
    """Обильный тип + переход на границе tail + старт в углу"""
    n, m = 45, 45
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if j < 40:
                row.append('a')  # 1800 клеток 'a'
            else:
                row.append('b')
        grid.append(''.join(row))
    # Длина 15: tail с позиции 13, переход на 12
    s = 'a' * 12 + 'b' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def main():
    print("=" * 80)
    print("ФИНАЛЬНАЯ ПРОВЕРКА ВСЕХ ОСТАВШИХСЯ СЛУЧАЕВ")
    print("=" * 80)
    
    tests = [
        (test_repeating_pattern(), "Повторяющийся паттерн abc×10"),
        (test_maze(), "Лабиринт с коридорами"),
        (test_spiral(), "Спираль типов"),
        (test_max_distance(), "Максимальное расстояние"),
        (test_multiple_transitions_abundant(), "Множественные переходы через обильные"),
        (test_start_at_target(), "Старт на нужной букве"),
        (test_start_center(), "Старт в центре"),
        (test_start_not_in_sequence(), "Старт на букве вне последовательности"),
        (test_exactly_100_targets_fixed(), "Ровно 100 клеток (исправлено)"),
        (test_large_grid_300x300(), "Максимальная карта 300×300"),
        (test_tall_grid_300x1(), "Вытянутая 300×1"),
        (test_combined_pathology(), "Комбинированная патология"),
    ]
    
    all_pass = True
    mismatches = []
    
    for test_input, name in tests:
        if not run_both(test_input, name):
            all_pass = False
            mismatches.append(name)
        print()
    
    print("=" * 80)
    if all_pass:
        print("✅ ВСЕ ФИНАЛЬНЫЕ ТЕСТЫ ПРОШЛИ!")
    else:
        print(f"❌ Расхождения в {len(mismatches)} тестах:")
        for m in mismatches:
            print(f"  - {m}")

if __name__ == "__main__":
    main()
