import time
import subprocess
import sys

# Тест на длинную последовательность доставок
print("Создаю тест с длинной последовательностью (200 доставок)...")

# 100x100 сетка с разными буквами
test_long_seq = "100 100\n50 50\n"
for i in range(100):
    row = ""
    for j in range(100):
        row += chr(ord('a') + ((i + j) % 26))
    test_long_seq += row + "\n"

# 200 доставок разных букв
test_long_seq += ("abcdefghijklmnopqrstuvwxyz" * 7 + "abcdefghijklmnopqr")

print(f"Последовательность длиной: {len('abcdefghijklmnopqrstuvwxyz' * 7 + 'abcdefghijklmnopqr')}")
print("Запускаю тест...")

start = time.time()
try:
    result = subprocess.run(
        [sys.executable, "solve_task3.py"],
        input=test_long_seq,
        capture_output=True,
        text=True,
        timeout=15
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"✓ Результат: {result.stdout.strip()}")
        print(f"✓ Время: {elapsed:.3f} сек")
        if elapsed < 3:
            print("✓✓ Отличная скорость для длинной последовательности!")
        elif elapsed < 5:
            print("✓✓ Хорошая скорость для длинной последовательности!")
    else:
        print(f"✗ Ошибка: {result.stderr}")
        
except subprocess.TimeoutExpired:
    elapsed = time.time() - start
    print(f"✗ TIMEOUT ({elapsed:.3f} сек)")
except Exception as e:
    print(f"✗ Исключение: {e}")
