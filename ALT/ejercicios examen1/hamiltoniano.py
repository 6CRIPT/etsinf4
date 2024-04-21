def ejer3(G):
    sol = [0]
    def backtracking():
        if len(sol) == len(G):
            if sol[0] in G[sol[-1]]:
                yield sol + [sol[0]]
        else:
            for v in range(len(G)):
                if v in G[sol[-1]] and v not in sol:
                        sol.append(v)
                        yield from backtracking()
                        sol.pop()
    yield from backtracking()
G = [[1,2,3], # del vertice 0 vamos a los vertices 1,2,3
    [0,3,4], # del vertice 1
    [0,3,5], # del vertice 2
    [0,1,2,4,5,6], # del vertice 3
    [1,3,7], # del vertice 4
    [2,3,6,8], # del vertice 5
    [3,5,7,8,9], # del vertice 6
    [4,6,9], # del vertice 7
    [5,6,9], # del vertice 8
    [6,7,8]] # del vertice 9

for solution in ejer3(G):
    print(solution)