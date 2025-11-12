"""
Comprehensive testing of all Task 3 solutions
"""
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple, List

def run_solution(script_path: str, test_input: str, timeout: int = 15) -> Tuple[str, float, str]:
    """
    Запускает решение и возвращает (результат, время, статус)
    """
    try:
        start = time.time()
        result = subprocess.run(
            ['python', script_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            return result.stdout.strip(), elapsed, "OK"
        else:
            return f"ERROR: {result.stderr[:100]}", elapsed, "ERROR"
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout, "TIMEOUT"
    except Exception as e:
        return f"EXCEPTION: {str(e)[:100]}", 0, "EXCEPTION"


def main():
    base_dir = Path(__file__).parent

    # Решения для сравнения
    solutions = {
        'Original (solve_task3.py)': base_dir / 'solve_task3.py',
        'Reference (no heuristics)': base_dir / 'solve_task3_reference.py',
        'Codex (multi-source Dijkstra)': base_dir / 'solve_task3_codex.py',
        'Claude (hybrid)': base_dir / 'solve_task3_claude.py',
    }

    # Проверяем существование файлов
    available_solutions = {}
    for name, path in solutions.items():
        if path.exists():
            available_solutions[name] = path
        else:
            print(f"⚠️  {name} не найден: {path}")

    if not available_solutions:
        print("❌ Ни одно решение не найдено!")
        return

    # Тесты для запуска
    test_files = [
        # Примеры из условия
        'test3_ex1.txt',
        'test3_ex2.txt',

        # Базовые тесты
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

        # Killer тесты
        'test_killer1_bfs_cap.txt',
        'test_killer2_max_states.txt',
        'test_killer3_lookahead.txt',

        # Codex тесты
        'test_far_cluster.txt',
        'test_long_alternating.txt',

        # Stress тесты
        'stress_test2.txt',
        'stress_test4.txt',
    ]

    print("=" * 120)
    print("COMPREHENSIVE TESTING OF TASK 3 SOLUTIONS")
    print("=" * 120)
    print()

    results: Dict[str, Dict[str, Tuple[str, float, str]]] = {}

    for test_file in test_files:
        test_path = base_dir / test_file
        if not test_path.exists():
            continue

        print(f"\n{'=' * 120}")
        print(f"Test: {test_file}")
        print(f"{'=' * 120}")

        test_input = test_path.read_text()

        for sol_name, sol_path in available_solutions.items():
            if sol_name not in results:
                results[sol_name] = {}

            answer, elapsed, status = run_solution(str(sol_path), test_input)
            results[sol_name][test_file] = (answer, elapsed, status)

            status_symbol = {
                'OK': '✓',
                'ERROR': '✗',
                'TIMEOUT': '⏱',
                'EXCEPTION': '💥'
            }.get(status, '?')

            print(f"  {status_symbol} {sol_name:40s}: {answer:10s} ({elapsed:.3f}s) [{status}]")

        # Проверяем согласованность результатов
        answers = set()
        for sol_name in available_solutions:
            if test_file in results[sol_name]:
                ans, _, status = results[sol_name][test_file]
                if status == 'OK':
                    answers.add(ans)

        if len(answers) > 1:
            print(f"  ⚠️  ВНИМАНИЕ: Расхождение в ответах: {answers}")
        elif len(answers) == 1:
            print(f"  ✓ Все решения согласованы: {list(answers)[0]}")

    # Итоговая статистика
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()

    for sol_name in available_solutions:
        ok_count = sum(1 for _, _, status in results[sol_name].values() if status == 'OK')
        timeout_count = sum(1 for _, _, status in results[sol_name].values() if status == 'TIMEOUT')
        error_count = sum(1 for _, _, status in results[sol_name].values() if status == 'ERROR')
        total_time = sum(t for _, t, status in results[sol_name].values() if status == 'OK')
        avg_time = total_time / ok_count if ok_count > 0 else 0

        print(f"{sol_name}:")
        print(f"  ✓ OK: {ok_count}")
        print(f"  ⏱ TIMEOUT: {timeout_count}")
        print(f"  ✗ ERROR: {error_count}")
        print(f"  ⏱ Average time: {avg_time:.3f}s")
        print(f"  ⏱ Total time: {total_time:.3f}s")
        print()

    # Сравнительная таблица
    print("=" * 120)
    print("COMPARATIVE ANALYSIS")
    print("=" * 120)
    print()

    # Находим расхождения
    mismatches: List[Tuple[str, Dict[str, str]]] = []
    for test_file in test_files:
        answers_by_solution = {}
        for sol_name in available_solutions:
            if test_file in results[sol_name]:
                ans, _, status = results[sol_name][test_file]
                if status == 'OK':
                    answers_by_solution[sol_name] = ans

        if len(set(answers_by_solution.values())) > 1:
            mismatches.append((test_file, answers_by_solution))

    if mismatches:
        print("❌ FOUND MISMATCHES:")
        print()
        for test_file, answers in mismatches:
            print(f"  Test: {test_file}")
            for sol_name, ans in answers.items():
                print(f"    {sol_name}: {ans}")
            print()
    else:
        print("✅ ALL SOLUTIONS AGREE ON ALL TESTS!")


if __name__ == "__main__":
    main()
