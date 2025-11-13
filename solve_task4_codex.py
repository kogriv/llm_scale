import math
import sys
sys.setrecursionlimit(10000)

def solve_case(n, adj, points):
    if n == 1:
        return [1]

    # Compute centroid to balance recursion
    parent = [-1] * n
    size = [0] * n
    order = []
    stack = [0]
    parent[0] = 0
    while stack:
        v = stack.pop()
        order.append(v)
        for u in adj[v]:
            if u == parent[v]:
                continue
            parent[u] = v
            stack.append(u)
    for v in reversed(order):
        s = 1
        for u in adj[v]:
            if u == parent[v]:
                continue
            s += size[u]
        size[v] = s
    centroid = 0
    best = n + 1
    for v in range(n):
        largest = n - size[v]
        for u in adj[v]:
            if parent[u] == v:
                largest = max(largest, size[u])
        if largest < best:
            best = largest
            centroid = v

    # Re-root at centroid to get subtree sizes with respect to centroid
    parent = [-1] * n
    size = [0] * n
    order = []
    stack = [centroid]
    parent[centroid] = centroid
    while stack:
        v = stack.pop()
        order.append(v)
        for u in adj[v]:
            if u == parent[v]:
                continue
            parent[u] = v
            stack.append(u)
    for v in reversed(order):
        s = 1
        for u in adj[v]:
            if u == parent[v]:
                continue
            s += size[u]
        size[v] = s

    # Prepare data structures
    coords = points
    assignment = [-1] * n

    def assign(v, p, available, anchor):
        assignment[v] = anchor
        if len(available) == 1:
            return
        anchor_point = coords[anchor]
        remaining = [idx for idx in available if idx != anchor]
        if not remaining:
            return
        children = [u for u in adj[v] if u != p]
        if not children:
            return
        # Sort children by subtree size (descending) to give larger blocks earlier
        children.sort(key=lambda node: size[node], reverse=True)

        def angle(idx):
            dx = coords[idx][0] - anchor_point[0]
            dy = coords[idx][1] - anchor_point[1]
            return math.atan2(dy, dx)

        def dist2(idx):
            dx = coords[idx][0] - anchor_point[0]
            dy = coords[idx][1] - anchor_point[1]
            return dx * dx + dy * dy

        remaining.sort(key=lambda idx: (angle(idx), dist2(idx)))
        ptr = 0
        for child in children:
            need = size[child]
            block = remaining[ptr:ptr + need]
            ptr += need
            if not block:
                continue
            child_anchor = min(block, key=lambda idx: (coords[idx][1], coords[idx][0]))
            assign(child, v, block, child_anchor)

    all_points = list(range(n))
    root_anchor = min(all_points, key=lambda idx: (coords[idx][1], coords[idx][0]))
    assign(centroid, -1, all_points, root_anchor)

    result = [0] * n
    for v in range(n):
        result[v] = assignment[v] + 1
    return result

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    t = int(next(it))
    out_lines = []
    for _ in range(t):
        n = int(next(it))
        adj = [[] for _ in range(n)]
        for _ in range(n - 1):
            u = int(next(it)) - 1
            v = int(next(it)) - 1
            adj[u].append(v)
            adj[v].append(u)
        points = []
        for _ in range(n):
            x = float(next(it))
            y = float(next(it))
            points.append((x, y))
        res = solve_case(n, adj, points)
        out_lines.append(' '.join(map(str, res)))
    sys.stdout.write('\n'.join(out_lines))

if __name__ == "__main__":
    main()
