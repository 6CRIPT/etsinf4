# AUTORES:
#César Martínez Chico
#Jaime Candel Martínez
# (poner aquí el nombre o 2 nombres del equipo de prácticas

def exact_cover(listaConjuntos, U=None):

    # permitimos que nos pasen U, si no lo hacen lo calculamos:
    if U is None:
        U = set().union(*listaConjuntos) 
    
    def backtracking(sol, cjtAcumulado):
        if len(sol) == len(listaConjuntos):
            if len(cjtAcumulado) == len(U):
                yield [c for c,s in zip(listaConjuntos,sol)if s == 1]
        else:
            cjt = listaConjuntos[len(sol)] #ESTO ES PARA ITERAR LOS CONJUNTOS INICIALES
            #LA IDEA ES QUE SI SI SON DISJUNTOS, ENTONCES METEMOS UN 1 AL ARRAY SOLUCION
            #Y LE PASAMOS A BACKTRACKING EL CONJUNTO ACUMULADO
            #  (O SEA TODOS LOS TERMINOS QUE SE VAN METIENDO)
            # CONCATENANDOLE EL CONJUNTO QUE ACABAMOS DE METER EL 1 EN SOL, ES DECIR,
            #EL QUE ESTAMOS ITERANDO
            if cjt.isdisjoint(cjtAcumulado):
                sol.append(1)
                yield from backtracking(sol, cjtAcumulado | cjt)
                sol.pop() #AHORA HAY QUE QUITAR EL ULTIMO
                #PORQUE ESTO EXPLORA UN HIJO HASTA EL FINAL
                #ENTONCES SI LLEGAMOS AL FINAL, HAY QUE VACIAR EL ARRAY 
                #RECUERDA QUE BACKTRACKING HACE RECURSION ENTONCES TENDRIAMOS EL
                #ARRAY LLENANDOSE HASTA EL FINAL, LUEGO LO VACIAMOS
                #Y SEGUIMOS CON EL SIGUIENTE, CON EL MISMO COSTE ESPACIAL.
            
            #METERLO TIENE LA CONDICION DE QUE SEAN DISJUNTOS
            #PERO NO METERLO SE HACE SIEMPRE.
            sol.append(0) #TODO ESTO ES ANALOGO AL CASO DISJUNTO
            yield from backtracking(sol, cjtAcumulado)
            #ESTO TERMINA CUANDO
            sol.pop()

    yield from backtracking([], set())

if __name__ == "__main__":
    listaConjuntos = [{"casa","coche","gato"},
                      {"casa","bici"},
                      {"bici","perro"},
                      {"boli","gato"},
                      {"coche","gato","bici"},
                      {"casa", "moto"},
                      {"perro", "boli"},
                      {"coche","moto"},
                      {"casa"}]
    for solucion in exact_cover(listaConjuntos):
        print(solucion)
