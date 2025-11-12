"""
Генератор стресс-тестов для выявления TLE
"""

# Тест 1: Максимальный размер карты с одним типом букв везде
print("Генерация теста 1: большая карта с одной буквой...")
with open("stress_test1.txt", 'w') as f:
    n, m = 300, 300
    f.write(f"{n} {m}\n")
    f.write(f"1 1\n")
    for i in range(n):
        f.write("a" * m + "\n")
    f.write("a" * 300 + "\n")  # 300 доставок
print("stress_test1.txt создан: 300x300, все 'a', 300 доставок 'a'")

# Тест 2: Максимальная последовательность доставок
print("\nГенерация теста 2: длинная последовательность...")
with open("stress_test2.txt", 'w') as f:
    n, m = 10, 10
    f.write(f"{n} {m}\n")
    f.write(f"1 1\n")
    # Разные буквы на карте
    letters = "abcdefghij"
    for i in range(n):
        f.write(letters + "\n")
    # Максимальная последовательность - 300 символов
    f.write("abcdefghij" * 30 + "\n")
print("stress_test2.txt создан: 10x10, 300 доставок разных букв")

# Тест 3: Много клеток одного типа
print("\nГенерация теста 3: много клеток одного типа...")
with open("stress_test3.txt", 'w') as f:
    n, m = 100, 100
    f.write(f"{n} {m}\n")
    f.write(f"50 50\n")
    # Половина карты - 'a', половина - 'b'
    for i in range(n):
        if i < n // 2:
            f.write("a" * m + "\n")
        else:
            f.write("b" * m + "\n")
    # Чередование a и b
    f.write("ab" * 150 + "\n")
print("stress_test3.txt создан: 100x100, чередование a/b")

# Тест 4: Много разных типов букв, разбросанных по карте
print("\nГенерация теста 4: все 26 букв...")
with open("stress_test4.txt", 'w') as f:
    n, m = 26, 26
    f.write(f"{n} {m}\n")
    f.write(f"13 13\n")
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for i in range(n):
        line = ""
        for j in range(m):
            line += alphabet[(i + j) % 26]
        f.write(line + "\n")
    # Последовательность всех букв несколько раз
    f.write(alphabet * 11 + "abcd" + "\n")  # 26*11 + 4 = 290
print("stress_test4.txt создан: 26x26, все буквы алфавита")

# Тест 5: Худший случай - зигзаг по большой карте
print("\nГенерация теста 5: зигзаг...")
with open("stress_test5.txt", 'w') as f:
    n, m = 150, 150
    f.write(f"{n} {m}\n")
    f.write(f"1 1\n")
    # 'a' только в углах
    for i in range(n):
        line = ""
        for j in range(m):
            if (i == 0 and j == 0) or (i == 0 and j == m-1) or \
               (i == n-1 and j == 0) or (i == n-1 and j == m-1):
                line += "a"
            else:
                line += "b"
        f.write(line + "\n")
    # Много доставок 'a' (должны выбирать между углами)
    f.write("a" * 100 + "\n")
print("stress_test5.txt создан: 150x150, 'a' только в углах")

print("\nВсе стресс-тесты созданы!")
