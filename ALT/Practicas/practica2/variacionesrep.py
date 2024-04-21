# AUTORES:
#César Martínez Chico
#Jaime Candel Martínez
# (poner aquí el nombre o 2 nombres del equipo de prácticas

#ACTIVIDAD 1
def variacionesRepeticion(elementos, cantidad):
    
    def backtracking(sol):
        if len(sol) == cantidad:
            yield sol.copy()
        else:
            for opcion in elementos:
                sol.append(opcion)
                yield from backtracking(sol)
                sol.pop()
                
    yield from backtracking([])

# COMPLETAR las actividades 1 y 2 para permutaciones y combinaciones

def permutaciones(elementos):
    
    def backtracking(sol):
        if len(sol) == len(elementos):
            yield sol.copy()
        else:
            for opcion in elementos:
                if(opcion not in sol):
                    sol.append(opcion)
                    yield from backtracking(sol)
                    sol.pop()
                
    yield from backtracking([])

def combinaciones(elementos, cantidad):
    
    def backtracking(sol, indice):
        if len(sol) == cantidad:
            yield sol.copy()
        else:
            for i in range(indice,len(elementos)):
                sol.append(elementos[i])
                yield from backtracking(sol, i + 1)
                sol.pop()
                
    yield from backtracking([],0)

if __name__ == "__main__":
    for n in (1,2,3):
        print('Variaciones con repeticion n =',n)
        for x in combinaciones(['tomate','queso','anchoas','aceitunas'],3):
            print(x)

    # probar las actividades 1 y 2 para permutaciones y combinaciones
