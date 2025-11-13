import sys

data = sys.stdin.read().strip().split()
it = iter(data)
n = int(next(it))
m = int(next(it))
sx = int(next(it))
sy = int(next(it))

print(f"n={n}, m={m}, sx={sx}, sy={sy}")

grid = []
for i in range(n):
    row = next(it)
    grid.append(row)
    print(f"row{i}: {row}")

non_del = next(it)
print(f"non_deliverable: {non_del}")

seq = next(it)
print(f"sequence: {seq}")
