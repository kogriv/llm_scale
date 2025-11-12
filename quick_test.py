import time
import subprocess
import sys

# Очень простые тесты
test_cases = []

# Тест 1: Минимальный (10x10, 5 доставок)
test_cases.append(("tiny", """10 10
1 1
""" + "a" * 10 + "\n" + ("a" * 10 + "\n") * 9 + "aaaaa"))

# Тест 2: 20x20, 10 доставок
test_cases.append(("small", """20 20
1 1
""" + "a" * 20 + "\n" + ("a" * 20 + "\n") * 19 + "a" * 10))

# Тест 3: 30x30, 15 доставок
test_cases.append(("medium_small", """30 30
1 1
""" + "a" * 30 + "\n" + ("a" * 30 + "\n") * 29 + "a" * 15))

# Тест 4: 50x50, 25 доставок
test_cases.append(("medium", """50 50
1 1
""" + "a" * 50 + "\n" + ("a" * 50 + "\n") * 49 + "a" * 25))

for name, test_input in test_cases:
    print(f"{name:15} ", end="", flush=True)
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "solve_task3.py"],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=10
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ {elapsed:6.3f}s  result={result.stdout.strip()}")
        else:
            print(f"✗ ERROR")
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT (>10s)")
    except Exception as e:
        print(f"✗ {e}")
