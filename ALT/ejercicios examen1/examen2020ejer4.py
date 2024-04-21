def ejer4(seq,V):
    sol = []
    def backtracking(indice, sumaparcial):
        if len(sol) == len(seq):
            if sum(seq[i] for i in range(len(sol)) if sol[i] == 1) == V:
                yield sol.copy()
        else:
            if indice <= len(seq):
                for num in seq[indice::]:
                    if sumaparcial + num <= V:
                        sol.append(1)
                        yield from backtracking(indice + 1, sumaparcial + num)
                        sol.pop()
                    else:
                        sol.append(0)
                        yield from backtracking(indice + 1, sumaparcial)
                        sol.pop()
            
            
    yield from backtracking(0,0)
for solution in ejer4([20,30,3,20,39,3,1,10,2], 24):
    print(solution)
