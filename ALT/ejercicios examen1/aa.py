def hamiltonian_cycle(G):
    def backtracking(path):
        if len(path)==len(G):
            if path[0] in G[path[-1]]: return path+[0]
        else:
            for v in [x for x in G[path[-1]] if x not in path]:
                found = backtracking(path+[v])
                if found!=None: return found
        return None
    return backtracking([0])
# ejemplo de uso:
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
print (hamiltonian_cycle(G))