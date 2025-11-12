"""
Комплексное стресс-тестирование с патологическими случаями
"""
import subprocess
import random
import string
from pathlib import Path
import time

def run_solution(script_path, test_input):
    """Запускает решение и возвращает (результат, время)"""
    try:
        start = time.time()
        result = subprocess.run(
            ['python', script_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=15
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            return result.stdout.strip(), elapsed
        else:
            return f"ERROR", elapsed
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 15.0
    except Exception as e:
        return f"EXCEPTION: {e}", 0

def generate_test(name, n, m, sx, sy, grid, s):
    """Формирует тестовый ввод"""
    lines = [f"{n} {m}", f"{sx} {sy}"]
    lines.extend(grid)
    lines.append(s)
    return "\n".join(lines) + "\n", name

def test_case_1():
    """Все клетки одного типа"""
    n, m = 20, 20
    grid = ['a' * m for _ in range(n)]
    s = 'a' * 15
    return generate_test("Все клетки 'a'", n, m, 1, 1, grid, s)

def test_case_2():
    """Много одного типа слева, редкий справа"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        row = 'a' * (m - 1) + 'b'
        grid.append(row)
    s = 'a' * 10 + 'b' + 'a' * 5
    return generate_test("Слева 'a', справа 'b'", n, m, 1, 1, grid, s)

def test_case_3():
    """Зигзаг: оптимум требует движения туда-сюда"""
    n, m = 25, 25
    grid = []
    for i in range(n):
        if i % 2 == 0:
            row = 'a' * (m // 2) + 'b' * (m - m // 2)
        else:
            row = 'b' * (m // 2) + 'a' * (m - m // 2)
        grid.append(row)
    s = 'ababababab'
    return generate_test("Зигзаг a-b", n, m, 1, 1, grid, s)

def test_case_4():
    """Один тип в углах, редкий в центре"""
    n, m = 40, 40
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if 15 <= i < 25 and 15 <= j < 25:
                row.append('b')
            else:
                row.append('a')
        grid.append(''.join(row))
    s = 'a' * 8 + 'b' + 'a' * 6
    return generate_test("'a' в углах, 'b' в центре", n, m, 1, 1, grid, s)

def test_case_5():
    """Длинная последовательность с повторениями"""
    n, m = 35, 35
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(chr(ord('a') + (i + j) % 10))
        grid.append(''.join(row))
    s = 'abcdefghij' * 5
    return generate_test("Длинная последовательность", n, m, 1, 1, grid, s)

def test_case_6():
    """Большой кластер в начале, маленький в конце"""
    n, m = 50, 50
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i < 45:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 12 + 'b' + 'a'
    return generate_test("Большой 'a', маленький 'b'", n, m, 1, 1, grid, s)

def test_case_7():
    """Множество редких типов"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            # Каждый тип встречается редко
            row.append(chr(ord('a') + ((i * m + j) % 26)))
        grid.append(''.join(row))
    s = 'abcdefghijklmnop'
    return generate_test("Множество редких типов", n, m, 1, 1, grid, s)

def test_case_8():
    """Переход между обильными типами"""
    n, m = 45, 45
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if j < m // 2:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 7 + 'b' * 7
    return generate_test("Переход a->b (оба обильны)", n, m, 1, 1, grid, s)

def test_case_9():
    """Старт далеко от всех целей"""
    n, m = 40, 40
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i < 5 or i >= 35 or j < 5 or j >= 35:
                row.append('a')  # По краям
            else:
                row.append('z')  # В центре — пустые клетки
        grid.append(''.join(row))
    s = 'a' * 10
    return generate_test("Старт в центре, цели по краям", n, m, 20, 20, grid, s)

def test_case_10():
    """Много переходов между типами"""
    n, m = 35, 35
    grid = []
    # Создаём полосы разных типов
    for i in range(n):
        row = []
        for j in range(m):
            type_idx = (i // 5) % 5
            row.append(chr(ord('a') + type_idx))
        grid.append(''.join(row))
    s = 'abcdea' * 3
    return generate_test("Много переходов", n, m, 1, 1, grid, s)

def test_case_11():
    """Очень обильный тип (>1000 клеток)"""
    n, m = 60, 60
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if (i + j) % 20 == 0:
                row.append('b')
            else:
                row.append('a')
        grid.append(''.join(row))
    s = 'a' * 10 + 'b' + 'a' * 5
    return generate_test("Очень обильный 'a' (>3000)", n, m, 1, 1, grid, s)

def test_case_12():
    """Последовательность с возвратом к началу"""
    n, m = 30, 30
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == 0:
                row.append('a')
            elif i == n - 1:
                row.append('b')
            else:
                row.append('c')
        grid.append(''.join(row))
    s = 'abcba'
    return generate_test("Возврат a->b->c->b->a", n, m, 1, 1, grid, s)

def test_case_13():
    """Равномерное распределение, много клеток каждого типа"""
    n, m = 52, 52  # 2704 клеток
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            # Каждая буква встречается ~104 раза
            row.append(chr(ord('a') + ((i * m + j) % 26)))
        grid.append(''.join(row))
    s = 'abcdefghijklmnopqrstuvwxyz'[:15]
    return generate_test("Равномерное распределение", n, m, 1, 1, grid, s)

def test_case_14():
    """Критический случай: переход на пороге обильности"""
    n, m = 25, 25  # 625 клеток
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            # 'a' встречается ровно 500 раз (на пороге), 'b' — 125 раз
            if (i * m + j) < 500:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 8 + 'b' + 'a' * 3
    return generate_test("Ровно 500 клеток 'a' (порог)", n, m, 1, 1, grid, s)

def test_case_15():
    """Чуть ниже порога обильности"""
    n, m = 22, 22  # 484 клеток
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            # 'a' встречается 480 раз (чуть ниже порога 500)
            if (i * m + j) < 480:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'a' * 8 + 'b' + 'a' * 3
    return generate_test("480 клеток 'a' (чуть ниже 500)", n, m, 1, 1, grid, s)

def main():
    base_dir = Path(__file__).parent
    optimized = base_dir / 'solve_task3.py'
    reference = base_dir / 'solve_task3_reference.py'
    
    test_cases = [
        test_case_1(), test_case_2(), test_case_3(), test_case_4(),
        test_case_5(), test_case_6(), test_case_7(), test_case_8(),
        test_case_9(), test_case_10(), test_case_11(), test_case_12(),
        test_case_13(), test_case_14(), test_case_15()
    ]
    
    print("=" * 80)
    print("КОМПЛЕКСНОЕ СТРЕСС-ТЕСТИРОВАНИЕ")
    print("=" * 80)
    
    mismatches = []
    timeouts = []
    
    for test_input, name in test_cases:
        print(f"\n{name}:")
        
        result_opt, time_opt = run_solution(optimized, test_input)
        result_ref, time_ref = run_solution(reference, test_input)
        
        print(f"  Оптимизированный: {result_opt} ({time_opt:.3f}s)")
        print(f"  Эталон:           {result_ref} ({time_ref:.3f}s)")
        
        if result_opt == "TIMEOUT":
            timeouts.append(name)
            print(f"  ⚠️  TIMEOUT на оптимизированном!")
        elif result_ref == "TIMEOUT":
            print(f"  ⚠️  TIMEOUT на эталоне (тест слишком сложный)")
        elif result_opt != result_ref:
            print(f"  ❌ РАСХОЖДЕНИЕ!")
            mismatches.append({
                'name': name,
                'test': test_input,
                'optimized': result_opt,
                'reference': result_ref
            })
        else:
            speedup = time_ref / time_opt if time_opt > 0 else 1
            print(f"  ✓ Совпадают (ускорение {speedup:.2f}x)")
    
    print("\n" + "=" * 80)
    print("ИТОГИ:")
    print("=" * 80)
    
    if mismatches:
        print(f"\n❌ Найдено {len(mismatches)} расхождений:")
        for m in mismatches:
            print(f"\n{'='*60}")
            print(f"Тест: {m['name']}")
            print(f"Оптимизированный: {m['optimized']}")
            print(f"Эталон (правильный): {m['reference']}")
            print(f"\nВходные данные:")
            print(m['test'][:500] + "..." if len(m['test']) > 500 else m['test'])
    
    if timeouts:
        print(f"\n⚠️  TIMEOUT на {len(timeouts)} тестах:")
        for t in timeouts:
            print(f"  - {t}")
    
    if not mismatches and not timeouts:
        print("\n✅ ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("Оптимизированное решение корректно на всех патологических случаях.")
    
    return len(mismatches) == 0 and len(timeouts) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
