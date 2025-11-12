import time
import subprocess
import sys

# Создаем тесты разной сложности
test_cases = []

# Тест 1: Малый (50x50, 10 доставок)
test_cases.append(("small", """50 50
1 1
""" + "a" * 50 + "\n" + ("a" * 50 + "\n") * 49 + "a" * 10))

# Тест 2: Средний (100x100, 20 доставок)
test_cases.append(("medium", """100 100
1 1
""" + "a" * 100 + "\n" + ("a" * 100 + "\n") * 99 + "a" * 20))

# Тест 3: Большой (150x150, 30 доставок)
test_cases.append(("large", """150 150
1 1
""" + "a" * 150 + "\n" + ("a" * 150 + "\n") * 149 + "a" * 30))

# Тест 4: Разнообразный (100x100, разные буквы)
grid = ""
for i in range(100):
    row = ""
    for j in range(100):
        row += chr(ord('a') + ((i + j) % 26))
    grid += row + "\n"
test_cases.append(("diverse", f"100 100\n1 1\n{grid}abcdefghij"))

# Тест 5: Много повторений (50x50, 50 доставок одной буквы)
test_cases.append(("repetitive", """50 50
1 1
""" + "a" * 50 + "\n" + ("a" * 50 + "\n") * 49 + "a" * 50))

for name, test_input in test_cases:
    print(f"\n{'='*50}")
    print(f"Тест: {name}")
    print(f"{'='*50}")
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "solve_task3.py"],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=5  # 5 секунд максимум
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ Результат: {result.stdout.strip()}")
            print(f"✓ Время: {elapsed:.3f} сек")
        else:
            print(f"✗ Ошибка: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"✗ TIMEOUT ({elapsed:.3f} сек)")
    except Exception as e:
        print(f"✗ Исключение: {e}")
