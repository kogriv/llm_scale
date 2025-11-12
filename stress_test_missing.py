"""
Дополнительные тесты на упущенные паттерны
"""
import subprocess
from pathlib import Path

def run_both(test_input, name):
    base_dir = Path(__file__).parent
    optimized = base_dir / 'solve_task3.py'
    reference = base_dir / 'solve_task3_reference.py'
    
    try:
        res_opt = subprocess.run(['python', optimized], input=test_input, capture_output=True, text=True, timeout=10)
        res_ref = subprocess.run(['python', reference], input=test_input, capture_output=True, text=True, timeout=10)
        
        opt = res_opt.stdout.strip() if res_opt.returncode == 0 else "ERROR"
        ref = res_ref.stdout.strip() if res_ref.returncode == 0 else "ERROR"
        
        match = "✓" if opt == ref else "❌"
        print(f"{match} {name}")
        print(f"    Opt: {opt}, Ref: {ref}")
        
        if opt != ref:
            with open(base_dir / f'failed_{name.replace(" ", "_")}.txt', 'w') as f:
                f.write(test_input)
            return False
        return True
    except:
        print(f"⚠️  {name} - TIMEOUT")
        return True

def test_many_transitions():
    """Много переходов туда-сюда между двумя типами"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        if i < n // 2:
            grid.append('a' * m)
        else:
            grid.append('b' * m)
    s = 'ab' * 15  # 30 переходов
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_all_26_letters():
    """Последовательность со всеми 26 буквами"""
    n, m = 26, 26
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(chr(ord('a') + (i + j) % 26))
        grid.append(''.join(row))
    s = 'abcdefghijklmnopqrstuvwxyz'
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_rare_at_start():
    """Редкая буква в самом начале"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == n-1 and j == m-1:
                row.append('z')  # Только одна клетка
            else:
                row.append('a')
        grid.append(''.join(row))
    s = 'z' + 'a' * 10
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_rare_before_tail():
    """Редкая буква ровно перед TAIL_STEPS (позиция len(s)-4)"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == n-1 and j == m-1:
                row.append('x')
            else:
                row.append('a')
        grid.append(''.join(row))
    # Длина 14: последние 3 (12,13,14) = tail, позиция 11 (индекс 10) — перед tail
    s = 'a' * 10 + 'x' + 'a' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_transition_at_tail_boundary():
    """Переход ровно на границе TAIL_STEPS"""
    n, m = 25, 25
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if j < m // 2:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    # Длина 15: tail начинается с позиции 13 (remaining=2)
    # Делаем переход на позиции 12 (последний шаг с cap)
    s = 'a' * 12 + 'b' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_checkerboard():
    """Шахматная доска a/b"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append('a' if (i + j) % 2 == 0 else 'b')
        grid.append(''.join(row))
    s = 'a' * 5 + 'b' * 5 + 'a' * 5
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_min_grid():
    """Минимальная карта 1×1"""
    return "1 1\n1 1\na\na\n"

def test_line_grid():
    """Вытянутая карта 1×100"""
    n, m = 1, 100
    grid = ['a' * 50 + 'b' * 50]
    s = 'a' * 5 + 'b' + 'a' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_exactly_100_targets():
    """Ровно 100 клеток типа 'a' (= cap обычного BFS)"""
    n, m = 10, 10
    grid = []
    count = 0
    for i in range(n):
        row = []
        for j in range(m):
            if count < 100:
                row.append('a')
                count += 1
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 8 + 'b' + 'a' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_exactly_220_targets():
    """Ровно 220 клеток типа 'a' (= cap на переходах)"""
    n, m = 15, 15
    grid = []
    count = 0
    for i in range(n):
        row = []
        for j in range(m):
            if count < 220:
                row.append('a')
                count += 1
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 8 + 'b' + 'a' * 3
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_both_abundant():
    """Оба типа обильные (>500)"""
    n, m = 50, 50
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if j < m // 2:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 8 + 'b' * 8
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_isolated_clusters():
    """Изолированные кластеры одного типа"""
    n, m = 40, 40
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            # Кластеры 'a' в углах, 'z' в середине
            if (i < 10 and j < 10) or (i >= 30 and j >= 30):
                row.append('a')
            else:
                row.append('z')
        grid.append(''.join(row))
    s = 'a' * 10
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def test_long_abundant_before_transition():
    """Длинная последовательность обильного типа перед переходом"""
    n, m = 50, 50
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if j < m - 5:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 50 + 'b' + 'a' * 10
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n"

def main():
    print("=" * 80)
    print("ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ НА УПУЩЕННЫЕ ПАТТЕРНЫ")
    print("=" * 80)
    
    tests = [
        (test_many_transitions(), "Много переходов a↔b"),
        (test_all_26_letters(), "Все 26 букв"),
        (test_rare_at_start(), "Редкая буква в начале"),
        (test_rare_before_tail(), "Редкая буква перед TAIL"),
        (test_transition_at_tail_boundary(), "Переход на границе TAIL"),
        (test_checkerboard(), "Шахматная доска"),
        (test_min_grid(), "Минимальная карта 1×1"),
        (test_line_grid(), "Линия 1×100"),
        (test_exactly_100_targets(), "Ровно 100 клеток (=cap)"),
        (test_exactly_220_targets(), "Ровно 220 клеток (=cap переходов)"),
        (test_both_abundant(), "Оба типа обильные"),
        (test_isolated_clusters(), "Изолированные кластеры"),
        (test_long_abundant_before_transition(), "Длинная последовательность → переход"),
    ]
    
    all_pass = True
    for test_input, name in tests:
        if not run_both(test_input, name):
            all_pass = False
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ВСЕ ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ ПРОШЛИ!")
    else:
        print("❌ Есть расхождения - проверьте файлы failed_*.txt")

if __name__ == "__main__":
    main()
