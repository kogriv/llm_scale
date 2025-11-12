"""Быстрая проверка корректности после изменения параметров"""
import subprocess
import time

def run_test(test_content, timeout=10):
    """Запуск теста с таймаутом"""
    try:
        start = time.time()
        result = subprocess.run(
            ['python', 'solve_task3.py'],
            input=test_content,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return int(output), elapsed
            else:
                return f"ERROR: Empty output", elapsed
        else:
            return f"ERROR: {result.stderr[:100]}", elapsed
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout
    except Exception as e:
        return f"ERROR: {e}", 0

def test_transition_abundant():
    """Тест с переходами между обильными типами"""
    n, m = 50, 50
    grid = []
    # Первая половина - 'a', вторая - 'b'
    for i in range(n):
        if i < 25:
            grid.append('a' * m)
        else:
            grid.append('b' * m)
    s = 'a' * 10 + 'b' * 10 + 'a' * 5
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test_threshold_boundary():
    """Тест ровно на границе порога 400"""
    n, m = 20, 20  # 400 клеток
    grid = ['a' * m for _ in range(n)]
    s = 'a' * 15
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

def test_tail_steps():
    """Тест для проверки хвостовых шагов"""
    n, m = 30, 30
    grid = []
    # Распределяем типы
    for i in range(n):
        row = ""
        for j in range(m):
            if (i + j) % 3 == 0:
                row += 'a'
            elif (i + j) % 3 == 1:
                row += 'b'
            else:
                row += 'c'
        grid.append(row)
    s = 'abc' * 8  # Последние 4 шага будут без cap
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\nz\n{s}\n"

print("🔍 Быстрая проверка после изменения параметров\n")

tests = [
    ("Переходы между обильными типами", test_transition_abundant()),
    ("Граница порога 400 клеток", test_threshold_boundary()),
    ("Проверка хвостовых шагов (TAIL_STEPS=4)", test_tail_steps()),
]

for name, test_content in tests:
    print(f"▶ {name}")
    result, elapsed = run_test(test_content)
    if result == "TIMEOUT":
        print(f"  ⚠️  TIMEOUT ({elapsed:.2f}s)")
    elif isinstance(result, str):
        print(f"  ❌ {result}")
    else:
        print(f"  ✅ Результат: {result} ({elapsed:.2f}s)")

print("\n✅ Базовая проверка завершена")
print("📊 Новые параметры:")
print("   - TAIL_STEPS = 4 (было 3)")
print("   - abundance_threshold = 400 (было 500)")
print("   - cap_normal = 150 (было 100)")
print("   - cap_on_transitions = 300 (было 220)")
