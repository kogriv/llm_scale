"""
Генератор патологических тестов для Task 3
"""
import random

def generate_abundant_transition_test():
    """
    Генерирует тест с:
    - Много клеток одного типа (например 'a')
    - Переход к редкому типу в середине последовательности
    - Оптимум требует "дальней" позиции 'a' перед переходом
    """
    n, m = 30, 30
    sx, sy = 1, 1
    
    # Создаём карту: левая половина — 'a', правая половина — тоже 'a' но с вкраплениями 'b'
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            if j < m // 2:
                row.append('a')  # Левая половина — только 'a'
            elif j == m - 1 and i == n // 2:
                row.append('b')  # Одна клетка 'b' справа
            else:
                row.append('a')  # Правая половина — тоже 'a'
        grid.append(''.join(row))
    
    # Последовательность: много 'a', потом 'b', потом снова 'a'
    # Оптимум: двигаться вправо по 'a', доставить 'b', остаться справа
    s = 'a' * 8 + 'b' + 'a' * 5
    
    print(f"{n} {m}")
    print(f"{sx} {sy}")
    for row in grid:
        print(row)
    print(s)

if __name__ == "__main__":
    generate_abundant_transition_test()
