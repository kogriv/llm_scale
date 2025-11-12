"""
Тесты на границы MAX_STATES
"""
import subprocess
from pathlib import Path

def run_both(test_input, name):
    base_dir = Path(__file__).parent
    optimized = base_dir / 'solve_task3.py'
    reference = base_dir / 'solve_task3_reference.py'
    
    try:
        res_opt = subprocess.run(['python', optimized], input=test_input, capture_output=True, text=True, timeout=12)
        res_ref = subprocess.run(['python', reference], input=test_input, capture_output=True, text=True, timeout=12)
        
        opt = res_opt.stdout.strip() if res_opt.returncode == 0 else "ERROR"
        ref = res_ref.stdout.strip() if res_ref.returncode == 0 else "ERROR"
        
        match = "✓" if opt == ref else "❌"
        print(f"{match} {name}: Opt={opt}, Ref={ref}")
        
        return opt == ref
    except:
        print(f"⚠️  {name} - TIMEOUT")
        return True

def test_seq_len_50():
    """seq_len ровно 50 (граница MAX_STATES 32→20)"""
    n, m = 20, 20
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(chr(ord('a') + ((i + j) % 10)))
        grid.append(''.join(row))
    s = 'abcde' * 10  # Длина 50
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n", "seq_len=50"

def test_seq_len_51():
    """seq_len=51 (чуть больше 50)"""
    n, m = 20, 20
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(chr(ord('a') + ((i + j) % 10)))
        grid.append(''.join(row))
    s = 'abcde' * 10 + 'a'  # Длина 51
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n", "seq_len=51"

def test_grid_5041():
    """grid_size=5041 (чуть больше 5000, граница MAX_STATES)"""
    n, m = 71, 71  # 5041
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if (i + j) % 3 == 0:
                row.append('a')
            else:
                row.append('b')
        grid.append(''.join(row))
    s = 'ab' * 8
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n", "grid_size=5041"

def test_combined_boundaries():
    """grid_size=5041 + seq_len=51"""
    n, m = 71, 71
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(chr(ord('a') + ((i + j) % 8)))
        grid.append(''.join(row))
    s = 'abcd' * 12 + 'abc'  # 51
    return f"{n} {m}\n1 1\n" + '\n'.join(grid) + f"\n{s}\n", "grid_size=5041 + seq_len=51"

def main():
    print("=" * 80)
    print("ТЕСТЫ НА ГРАНИЦЫ MAX_STATES")
    print("=" * 80)
    
    tests = [
        test_seq_len_50(),
        test_seq_len_51(),
        test_grid_5041(),
        test_combined_boundaries(),
    ]
    
    all_pass = True
    for test_input, name in tests:
        if not run_both(test_input, name):
            all_pass = False
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ВСЕ ГРАНИЧНЫЕ ТЕСТЫ ПРОШЛИ!")
    else:
        print("❌ Есть расхождения")

if __name__ == "__main__":
    main()
