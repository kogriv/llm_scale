"""
Тестирование граничных случаев вокруг порога обильности (500)
"""
import subprocess
from pathlib import Path

def run_solution(script_path, test_input):
    try:
        result = subprocess.run(
            ['python', script_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "ERROR"
    except:
        return "TIMEOUT"

def generate_threshold_test(num_targets_a, name):
    """
    Генерирует тест с точным количеством клеток типа 'a'
    """
    # Вычисляем размер карты
    total = num_targets_a + 100  # +100 для клеток 'b'
    n = int(total ** 0.5) + 1
    m = n
    
    grid = []
    count_a = 0
    for i in range(n):
        row = []
        for j in range(m):
            if count_a < num_targets_a:
                row.append('a')
                count_a += 1
            else:
                row.append('b')
        grid.append(''.join(row))
    
    # Последовательность: переходы между a и b
    s = 'a' * 8 + 'b' + 'a' * 5
    
    lines = [f"{n} {m}", "1 1"]
    lines.extend(grid)
    lines.append(s)
    return "\n".join(lines) + "\n", name

def main():
    base_dir = Path(__file__).parent
    optimized = base_dir / 'solve_task3.py'
    reference = base_dir / 'solve_task3_reference.py'
    
    # Тесты вокруг порога 500
    test_cases = [
        (490, "490 клеток 'a' (ниже порога)"),
        (495, "495 клеток 'a'"),
        (499, "499 клеток 'a' (на грани)"),
        (500, "500 клеток 'a' (ровно порог)"),
        (501, "501 клетка 'a' (чуть выше)"),
        (505, "505 клеток 'a'"),
        (510, "510 клеток 'a' (выше порога)"),
        (600, "600 клеток 'a' (старый порог)"),
        (650, "650 клеток 'a'"),
        (800, "800 клеток 'a'"),
    ]
    
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ГРАНИЧНЫХ СЛУЧАЕВ ПОРОГА ОБИЛЬНОСТИ (500)")
    print("=" * 80)
    
    mismatches = []
    
    for num_targets, name in test_cases:
        test_input, test_name = generate_threshold_test(num_targets, name)
        
        print(f"\n{test_name}:")
        
        result_opt = run_solution(optimized, test_input)
        result_ref = run_solution(reference, test_input)
        
        print(f"  Оптимизированный: {result_opt}")
        print(f"  Эталон:           {result_ref}")
        
        if result_opt != result_ref:
            print(f"  ❌ РАСХОЖДЕНИЕ!")
            mismatches.append({
                'name': test_name,
                'num_targets': num_targets,
                'optimized': result_opt,
                'reference': result_ref,
                'test': test_input
            })
        else:
            print(f"  ✓ Совпадают")
    
    print("\n" + "=" * 80)
    
    if mismatches:
        print(f"\n❌ Найдено {len(mismatches)} расхождений:")
        for m in mismatches:
            print(f"\n{m['name']} ({m['num_targets']} клеток):")
            print(f"  Оптимизированный: {m['optimized']}")
            print(f"  Эталон: {m['reference']}")
            
            # Сохраняем первое расхождение
            if len(mismatches) == 1:
                with open(base_dir / 'failed_threshold_test.txt', 'w') as f:
                    f.write(m['test'])
                print(f"  Тест сохранён в failed_threshold_test.txt")
    else:
        print("\n✅ ВСЕ ГРАНИЧНЫЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
    
    return len(mismatches) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
