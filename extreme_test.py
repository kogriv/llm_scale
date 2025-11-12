import time
import subprocess
import sys

# Экстремальный тест: 300x300, 300 доставок
print("Создаю тест 300x300 с 300 доставками...")

test_extreme = "300 300\n1 1\n"
# Сетка с разными буквами
for i in range(300):
    row = ""
    for j in range(300):
        row += chr(ord('a') + ((i + j) % 26))
    test_extreme += row + "\n"

# 300 доставок разных букв
test_extreme += "abcdefghijklmnopqrstuvwxyz" * 11 + "abcdefghijklmnopq"

print("Запускаю тест...")
start = time.time()
try:
    result = subprocess.run(
        [sys.executable, "solve_task3.py"],
        input=test_extreme,
        capture_output=True,
        text=True,
        timeout=15
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"✓ Результат: {result.stdout.strip()}")
        print(f"✓ Время: {elapsed:.3f} сек")
        if elapsed < 10:
            print("✓✓ Отличная скорость!")
    else:
        print(f"✗ Ошибка: {result.stderr}")
        
except subprocess.TimeoutExpired:
    elapsed = time.time() - start
    print(f"✗ TIMEOUT ({elapsed:.3f} сек)")
except Exception as e:
    print(f"✗ Исключение: {e}")
