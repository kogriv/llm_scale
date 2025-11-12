"""
Генератор тестовых случаев для task3
"""

# Тест 1: Все доставки в одну и ту же клетку (минимальное перемещение)
test1 = """1 1
1 1
a
aaa"""

# Тест 2: Одна доставка - уже в нужной клетке
test2 = """1 1
1 1
a
a"""

# Тест 3: Линейное перемещение
test3 = """1 5
1 1
abcde
abcde"""

# Тест 4: Зигзаг - несколько клеток одного типа
test4 = """3 3
1 1
abc
def
ghi
adg"""

# Тест 5: Выбор между двумя клетками одного типа
test5 = """3 3
2 2
aaa
aaa
aaa
aaa"""

# Тест 6: Большая карта с повторяющимися адресами
test6 = """5 5
1 1
abcde
fghij
klmno
pqrst
uvwxy
abcdefghijklmnopqrstuvwxy"""

# Тест 7: Возврат назад
test7 = """1 5
1 3
abcde
aba"""

# Тест 8: Все буквы на одной позиции (вырожденный случай)
test8 = """2 2
1 1
ab
cd
abcdabcd"""

# Тест 9: Максимальное расстояние
test9 = """3 3
1 1
abc
def
ghi
agi"""

# Тест 10: Одна строка, много букв
test10 = """1 10
1 1
abcdefghij
jihgfedcba"""

tests = [
    ("test1_same_cell", test1, "Все доставки в одну клетку"),
    ("test2_already_there", test2, "Уже в нужной клетке"),
    ("test3_linear", test3, "Линейное движение"),
    ("test4_zigzag", test4, "Зигзаг"),
    ("test5_multiple_same", test5, "Множество клеток одного типа"),
    ("test6_large", test6, "Большая карта"),
    ("test7_backtrack", test7, "Возврат назад"),
    ("test8_all_in_one", test8, "Разные буквы в одной области"),
    ("test9_max_dist", test9, "Максимальное расстояние"),
    ("test10_line", test10, "Одна строка"),
]

for name, content, description in tests:
    filename = f"{name}.txt"
    with open(filename, 'w') as f:
        f.write(content.strip())
    print(f"Создан {filename}: {description}")
