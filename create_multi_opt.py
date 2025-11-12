# Тест: оптимальный выбор при множественных вариантах
test_opt = """3 5
2 3
aaaaa
bbbbb
aaaaa
aba"""

with open("test_multi_opt.txt", 'w') as f:
    f.write(test_opt.strip())
    
print("Тест создан: test_multi_opt.txt")
print("Начало (2,3) - в ряду 'b'")
print("a: ближайшая 'a' в (1,3) или (3,3), dist=1")
print("b: уже в ряду 'b', можем остаться в (2,3) или пойти в другую, dist=0")
print("a: от (2,3) до 'a' = 1")
print("Ожидаемый результат: 1+0+1 = 2")
