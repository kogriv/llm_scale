"""
Более сложный патологический тест
Карта с большим количеством одного типа в разных концах
"""
def generate_complex_test():
    n, m = 50, 50
    sx, sy = 1, 1
    
    # Заполняем карту
    grid = []
    for i in range(n):
        row = []
        for j in range(m):
            # Создаём несколько зон 'a' и одну зону 'b'
            if i < 10 or i >= 40:
                row.append('a')  # Верх и низ — 'a'
            elif 20 <= i < 25 and 45 <= j < 50:
                row.append('b')  # Маленький кластер 'b' справа внизу
            else:
                # Середина — смесь других букв
                row.append(chr(ord('c') + ((i + j) % 20)))
        grid.append(''.join(row))
    
    # Последовательность: несколько 'a', потом переход к 'b', потом опять 'a'
    # Оптимум может требовать пройти к дальней 'a' внизу перед 'b'
    s = 'aaaaaaa' + 'b' + 'aaaa'
    
    print(f"{n} {m}")
    print(f"{sx} {sy}")
    for row in grid:
        print(row)
    print(s)

if __name__ == "__main__":
    generate_complex_test()
