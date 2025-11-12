"""
Сравнение solve_task3.py (оптимизированный) с solve_task3_reference.py (эталон)
"""
import subprocess
import sys
from pathlib import Path

def run_solution(script_path, test_path):
    """Запускает решение и возвращает результат"""
    try:
        result = subprocess.run(
            ['python', script_path],
            stdin=open(test_path, 'r'),
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"ERROR: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"EXCEPTION: {e}"

def main():
    base_dir = Path(__file__).parent
    optimized = base_dir / 'solve_task3.py'
    reference = base_dir / 'solve_task3_reference.py'
    
    # Список тестов для проверки
    test_files = [
        'test3_ex1.txt',
        'test3_ex2.txt',
        'test1_same_cell.txt',
        'test2_already_there.txt',
        'test3_linear.txt',
        'test4_zigzag.txt',
        'test5_multiple_same.txt',
        'test6_large.txt',
        'test7_backtrack.txt',
        'test8_all_in_one.txt',
        'test9_max_dist.txt',
        'test10_line.txt',
    ]
    
    print("Сравнение оптимизированного и эталонного решений")
    print("=" * 70)
    
    mismatches = []
    
    for test_file in test_files:
        test_path = base_dir / test_file
        if not test_path.exists():
            continue
            
        print(f"\nТест: {test_file}")
        
        result_opt = run_solution(optimized, test_path)
        result_ref = run_solution(reference, test_path)
        
        print(f"  Оптимизированный: {result_opt}")
        print(f"  Эталон:           {result_ref}")
        
        if result_opt != result_ref:
            print(f"  ❌ РАСХОЖДЕНИЕ!")
            mismatches.append({
                'test': test_file,
                'optimized': result_opt,
                'reference': result_ref
            })
        else:
            print(f"  ✓ Совпадают")
    
    print("\n" + "=" * 70)
    if mismatches:
        print(f"\n❌ Найдено {len(mismatches)} расхождений:")
        for m in mismatches:
            print(f"\nТест: {m['test']}")
            print(f"  Оптимизированный: {m['optimized']}")
            print(f"  Эталон (правильный): {m['reference']}")
    else:
        print("\n✓ Все тесты совпали! Оптимизированное решение корректно на доступных тестах.")

if __name__ == "__main__":
    main()
