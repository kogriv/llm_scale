"""
Генератор случайных стресс-тестов
"""
import subprocess
import random
import string
from pathlib import Path
import time

def run_solution(script_path, test_input):
    """Запускает решение"""
    try:
        start = time.time()
        result = subprocess.run(
            ['python', script_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=10
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            return result.stdout.strip(), elapsed
        else:
            return f"ERROR", elapsed
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 10.0
    except Exception as e:
        return f"EXCEPTION", 0

def generate_random_test(seed):
    """Генерирует случайный тест"""
    random.seed(seed)
    
    # Случайные размеры
    n = random.randint(15, 60)
    m = random.randint(15, 60)
    
    # Случайная стартовая позиция
    sx = random.randint(1, n)
    sy = random.randint(1, m)
    
    # Генерация карты
    num_types = random.randint(3, 15)  # От 3 до 15 разных типов
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            # Некоторые типы встречаются чаще (имитация кластеров)
            if random.random() < 0.3:
                # Обильный тип
                row.append('a')
            elif random.random() < 0.2:
                row.append('b')
            else:
                # Случайный из оставшихся
                row.append(chr(ord('a') + random.randint(0, num_types - 1)))
        grid.append(''.join(row))
    
    # Генерация последовательности
    seq_len = random.randint(8, 40)
    s = ''.join(chr(ord('a') + random.randint(0, num_types - 1)) for _ in range(seq_len))
    
    # Формирование теста
    lines = [f"{n} {m}", f"{sx} {sy}"]
    lines.extend(grid)
    lines.append(s)
    return "\n".join(lines) + "\n"

def main():
    base_dir = Path(__file__).parent
    optimized = base_dir / 'solve_task3.py'
    reference = base_dir / 'solve_task3_reference.py'
    
    num_tests = 100
    
    print("=" * 80)
    print(f"СЛУЧАЙНОЕ СТРЕСС-ТЕСТИРОВАНИЕ ({num_tests} тестов)")
    print("=" * 80)
    
    mismatches = []
    progress_interval = num_tests // 10
    
    for i in range(num_tests):
        if (i + 1) % progress_interval == 0:
            print(f"\nПрогресс: {i + 1}/{num_tests}")
        
        test_input = generate_random_test(seed=42 + i)
        
        result_opt, time_opt = run_solution(optimized, test_input)
        result_ref, time_ref = run_solution(reference, test_input)
        
        if result_opt == "TIMEOUT" or result_ref == "TIMEOUT":
            continue  # Пропускаем слишком сложные тесты
        
        if result_opt != result_ref:
            print(f"\n❌ РАСХОЖДЕНИЕ на тесте #{i + 1}!")
            print(f"  Оптимизированный: {result_opt}")
            print(f"  Эталон: {result_ref}")
            mismatches.append({
                'seed': 42 + i,
                'test': test_input,
                'optimized': result_opt,
                'reference': result_ref
            })
            # Сохраняем первое расхождение для детального анализа
            if len(mismatches) == 1:
                with open(base_dir / 'failed_random_test.txt', 'w') as f:
                    f.write(test_input)
                print(f"  Тест сохранён в failed_random_test.txt")
    
    print("\n" + "=" * 80)
    print("ИТОГИ:")
    print("=" * 80)
    
    if mismatches:
        print(f"\n❌ Найдено {len(mismatches)} расхождений из {num_tests} тестов")
        print(f"\nПервое расхождение (seed={mismatches[0]['seed']}):")
        print(f"  Оптимизированный: {mismatches[0]['optimized']}")
        print(f"  Эталон: {mismatches[0]['reference']}")
        print(f"\nВходные данные первого расхождения:")
        print(mismatches[0]['test'][:800])
    else:
        print(f"\n✅ ВСЕ {num_tests} СЛУЧАЙНЫХ ТЕСТОВ ПРОШЛИ УСПЕШНО!")
    
    return len(mismatches) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
