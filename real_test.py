import time
import subprocess
import sys

tests = [
    ("100x100_diverse", """100 100
50 50
""" + "\n".join("".join(chr(ord('a') + ((i+j) % 26)) for j in range(100)) for i in range(100)) + "\nabcdefghij"),
    
    ("100x100_same_20", """100 100
1 1
""" + "\n".join("a" * 100 for _ in range(100)) + "\n" + "a" * 20),
    
    ("150x150_diverse", """150 150
75 75
""" + "\n".join("".join(chr(ord('a') + ((i+j) % 26)) for j in range(150)) for i in range(150)) + "\nabcdefghijklmno"),
]

for name, test_input in tests:
    print(f"{name:20} ", end="", flush=True)
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
            print(f"✓ {elapsed:6.3f}s")
        else:
            print(f"✗ ERROR")
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT")
    except Exception as e:
        print(f"✗ {e}")
