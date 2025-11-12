"""
Проверка на оптимизацию выбора между несколькими клетками одного типа
"""

# Проблемный случай: несколько 'a' в разных местах
# Ровер должен выбрать оптимальную клетку для каждой доставки

test_optimization = """5 5
3 3
axxxa
xxxxx
xxaxx
xxxxx
axxxa
aba"""

with open("test_optimization.txt", 'w') as f:
    f.write(test_optimization.strip())

print("Создан test_optimization.txt")
print("\nОписание:")
print("Карта 5x5 с 'a' в углах и в центре")
print("Начало: (3,3) - центр, где есть 'a'")
print("Доставки: aba")
print("a: уже в (3,3), dist=0")
print("b: нет 'b' на карте - это проблема!")
