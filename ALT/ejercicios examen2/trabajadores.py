def voraz(matriz):
    sol = []
    for trabajador in matriz:
        cond = [elemento for elemento in trabajador if elemento not in sol]
        trabajo = min(cond)
        sol.append(trabajo)
    return sol
matriz = [[1,2],[1,2]]
print(voraz(matriz))