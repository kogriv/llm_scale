"""Тестирование solve_task3_claude.py на всех существующих тестах"""
import subprocess
import time
import os
from pathlib import Path

def run_test(test_file, solver="solve_task3_claude.py", timeout=15):
    """Запуск одного теста"""
    try:
        start = time.time()
        result = subprocess.run(
            ['python', solver],
            stdin=open(test_file, 'r'),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            output = result.stdout.strip()
            try:
                return int(output), elapsed, "OK"
            except ValueError:
                return None, elapsed, f"INVALID_OUTPUT: {output[:50]}"
        else:
            return None, elapsed, f"ERROR: {result.stderr[:100]}"
    except subprocess.TimeoutExpired:
        return None, timeout, "TIMEOUT"
    except FileNotFoundError:
        return None, 0, "FILE_NOT_FOUND"
    except Exception as e:
        return None, 0, f"EXCEPTION: {str(e)[:100]}"

def find_all_tests():
    """Найти все тестовые файлы"""
    test_files = []
    
    # Базовые тесты
    for pattern in ['test*.txt', 'stress_test*.txt']:
        test_files.extend(Path('.').glob(pattern))
    
    return sorted(set([str(f) for f in test_files]))

print("=" * 80)
print("ТЕСТИРОВАНИЕ: solve_task3_claude.py")
print("=" * 80)

test_files = find_all_tests()
print(f"\nНайдено тестов: {len(test_files)}\n")

results = []
total_time = 0
ok_count = 0
timeout_count = 0
error_count = 0

for i, test_file in enumerate(test_files, 1):
    print(f"[{i:2d}/{len(test_files)}] {test_file:40s} ", end='', flush=True)
    
    result, elapsed, status = run_test(test_file)
    total_time += elapsed
    
    if status == "OK":
        print(f"✅ {result:5d}  ({elapsed:6.3f}s)")
        ok_count += 1
        results.append((test_file, result, elapsed, "OK"))
    elif status == "TIMEOUT":
        print(f"⏱️  TIMEOUT (>{elapsed:.1f}s)")
        timeout_count += 1
        results.append((test_file, None, elapsed, "TIMEOUT"))
    else:
        print(f"❌ {status[:40]}")
        error_count += 1
        results.append((test_file, None, elapsed, status))

print("\n" + "=" * 80)
print("ИТОГОВАЯ СТАТИСТИКА")
print("=" * 80)
print(f"Всего тестов:     {len(test_files)}")
print(f"✅ Успешно:       {ok_count} ({100*ok_count/len(test_files):.1f}%)")
print(f"⏱️  Timeout:       {timeout_count}")
print(f"❌ Ошибки:        {error_count}")
print(f"Общее время:      {total_time:.3f}s")
if ok_count > 0:
    print(f"Среднее время:    {total_time/ok_count:.3f}s")
print()

# Группировка по категориям
print("=" * 80)
print("РЕЗУЛЬТАТЫ ПО КАТЕГОРИЯМ")
print("=" * 80)

categories = {
    'Примеры': [],
    'Базовые тесты': [],
    'Killer тесты': [],
    'Новые тесты (codex)': [],
    'Stress тесты': [],
}

for test_file, result, elapsed, status in results:
    if 'ex' in test_file:
        categories['Примеры'].append((test_file, result, elapsed, status))
    elif 'killer' in test_file:
        categories['Killer тесты'].append((test_file, result, elapsed, status))
    elif 'far_cluster' in test_file or 'long_alternating' in test_file:
        categories['Новые тесты (codex)'].append((test_file, result, elapsed, status))
    elif 'stress' in test_file:
        categories['Stress тесты'].append((test_file, result, elapsed, status))
    else:
        categories['Базовые тесты'].append((test_file, result, elapsed, status))

for category, tests in categories.items():
    if tests:
        print(f"\n{category}:")
        for test_file, result, elapsed, status in tests:
            if status == "OK":
                print(f"  ✅ {test_file:40s} → {result:5d} ({elapsed:.3f}s)")
            elif status == "TIMEOUT":
                print(f"  ⏱️  {test_file:40s} → TIMEOUT")
            else:
                print(f"  ❌ {test_file:40s} → {status[:30]}")

print("\n" + "=" * 80)
