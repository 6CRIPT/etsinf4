def quijote(C):
    #el final mas corto primero
    if len(C)>0:
        principio_inicial, final_previo = min(C, key= lambda x:x[1]) #el primero
        sol = [(principio_inicial, final_previo)] #metemos el primero
        for principio,final in sorted(C, key = lambda trozo: trozo[1]): #ordenamos teniendo en cuenta los finales
            if final_previo <= principio: #si el ultimo final es como maximo el principio, se pueden solapar
                sol.append((principio, final))
                final_previo = final
        return sol

C = [(23,40), (12,50), (4,8), (10,12), (20,25)] #cada tupla -> primer num es el parrafo que desean empezar, el segundo terminar.
print(quijote(C))

def quijote2(C):
    x = []
    t2 = min(s for (s,t) in C)
    for (s,t) in sorted(C, key = lambda x: x[1] ):
        if t2<=s:
            x.append((s,t))
            t2 = t
    return x
print(quijote2(C))
