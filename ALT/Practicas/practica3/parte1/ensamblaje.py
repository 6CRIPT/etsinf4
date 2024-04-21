import numpy as np
import heapq
import collections
import argparse
#HECHO POR CESAR MARTINEZ CHICO Y JAIME CANDEL MARTINEZ
######################################################################
#
#                     GENERACIÓN DE INSTANCIAS
#
######################################################################

def genera_instancia(M, low=1, high=4, seed=42):
    np.random.seed(seed)
    return np.random.randint(low=low,high=high,
                             size=(M,M),dtype=int)

######################################################################
#
#                       ALGORITMOS VORACES
#
######################################################################

def compute_score(costMatrix, solution):
    return sum(costMatrix[pieza,instante]
               for pieza,instante in enumerate(solution))

def naive_solution(costMatrix):
    solution = list(range(costMatrix.shape[0]))
    return compute_score(costMatrix,solution), solution

#EL INDICE ES LA PIEZA Y EL X INDICE ES EL INSTANTE EN EL QUE ESA PIEZA I TIENE COSTE MINIMO
def voraz_x_pieza(costMatrix):
    # costMatrix[i,j] el coste de situar pieza i en instante j
    M = costMatrix.shape[0] # nº piezas
    solution = []
    score = 0
    for pieza in range(M):
        min = np.inf
        instante_minimo = -1
        for instante in range(M):
            if instante not in solution:
                if costMatrix[pieza, instante] < min:
                    min = costMatrix[pieza,instante]
                    instante_minimo = instante
        solution.append(instante_minimo)
        score += min    

    return score,solution

def voraz_x_instante(costMatrix):
    # costMatrix[i,j] el coste de situar pieza i en instante j
    M = costMatrix.shape[0] # nº piezas
    solution = [-1] * M
    score = 0
    for instante in range(M): #*desempaqueta la matriz en filas, separandolas, al hacer zip, lo que haces es convertir las filas en iterables
        #si no recuerdas como funciona, pregunta a chatgpt solomente qué hace *matriz, pero vamos que desempaqueta en las 3 filas, y luego el zip
        #lo que hace es coger el primero de cada 3 filas, es decir, el primer elemento de cada columna, para poder iterar las columnas
        #pieza_min = min(range(M), key = lambda pieza: instante[pieza]) #recuerda que aqui lo que iteras es de 0 a M-1, QUE SON LAS PIEZAS, TE DEVUELVE LA PIEZA CON COSTE MINIMO, NO SU COSTE
        #solution[pieza_min] = instante
        #score += costMatrix[pieza_min, instante]
        pieza_min = -1
        min = np.inf
        for pieza in range(M):
            if pieza not in solution and costMatrix[pieza, instante] < min:
                min = costMatrix[pieza, instante]
                pieza_min = pieza
        solution[instante] = pieza_min
        score += min
    solution_def = [-1]*M
    for instante, pieza in enumerate(solution):
        solution_def[pieza] = instante     #darle la vuelta a la solucion voraz
    return score,solution_def

# def voraz_x_coste(costMatrix):
#     # costMatrix[i,j] el coste de situar pieza i en instante j
#     M = costMatrix.shape[0] # nº piezas
#     solution = [-1]*M
#     costes = []
#     for i in range(M):
#         min = float('inf')
#         for indice_pieza,pieza in enumerate(costMatrix):
#             if indice_pieza not in solution:
#                 for indice_instante, coste_instante in enumerate(pieza):
#                     if coste_instante < min:
#                         min = coste_instante
#                         posicion = (indice_pieza, indice_instante)
#             if solution[indice_instante] > coste_instante:
#                 solution[indice_instante] = indice_pieza
#     score = sum(costes)
#     return score,solution

def voraz_x_coste(costMatrix):
    # costMatrix[i,j] el coste de situar pieza i en instante j
    M = costMatrix.shape[0] # nº piezas
    solution = [-1]*M
    costes_ordenados = []
    score = 0
    for pieza in range(M):
        for instante in range(M):
            costes_ordenados.append((costMatrix[pieza,instante], pieza, instante))
    costes_ordenados = sorted(costes_ordenados) #ordenamos
    num = 0
    sol_aux = []
    while(num<M):
        coste, pieza, instante = costes_ordenados.pop(0)
        if solution[pieza] == -1 and instante not in solution:
            solution[pieza] = instante 
            score += coste
            num +=1
    return score,solution


def voraz_combina(costMatrix):
    
    score2, solution2 = voraz_x_pieza(costMatrix)
    score3, solution3 = voraz_x_instante(costMatrix)
    score4, solution4 = voraz_x_coste(costMatrix)

    solutions = {score2: solution2, score3: solution3, score4: solution4}
    score = min(solutions.keys())
    solution = solutions[score]

    return score,solution
        
######################################################################
#
#                       RAMIFICACIÓN Y PODA
#
######################################################################

class Ensamblaje:

    def __init__(self, costMatrix, initial_sol = None):
        '''
        costMatrix es una matriz numpy MxM con valores positivos
        costMatrix[i,j] es el coste de ensamblar la pieza i cuando ya
        se han ensamblado j piezas.
        '''
        # no haría falta pero por si acaso comprobamos que costMatrix
        # es una matriz numpy cuadrada y de costMatrix positivos
        assert(type(costMatrix) is np.ndarray and len(costMatrix.shape) == 2
               and costMatrix.shape[0] == costMatrix.shape[1]
               and costMatrix.dtype == int and costMatrix.min()>=0)
        self.costMatrix = costMatrix
        self.M = costMatrix.shape[0]
        # la forma más barata de ensamblar la pieza i si podemos
        # elegir el momento de ensamblaje que más nos convenga:
        self.minPieza = [costMatrix[i,:].min() for i in range(self.M)]
        self.x = initial_sol
        if initial_sol is None:
            self.fx = np.inf
        else:
            self.fx = compute_score(costMatrix,initial_sol)
        
    def branch(self, s_score, s):
        '''
        s_score es el score de s
        s es una solución parcial
        '''
        i = len(s) # i es la siguiente pieza a montar, i<M
        
        # costMatrix[i,j] coste ensamblar objeto i en instante j
        for j in range(self.M): # todos los instantes
            # si j no ha sido utilizado en s
            if j not in s: # NO es la forma más eficiente
                           # al ser lineal con len(s)
                new_score = s_score - self.minPieza[i] + self.costMatrix[i,j]
                yield (new_score, s + [j])

    def is_complete(self, s):
        '''
        s es una solución parcial
        '''
        return len(s) == self.M

    def initial_solution(self):
        return (sum(self.minPieza),[])

    def solve(self):
        A = [ self.initial_solution() ] # cola de prioridad
        iterations = 0 # nº iteraciones
        gen_states = 0 # nº estados generados
        podas_opt  = 0 # nº podas por cota optimista
        maxA       = 0 # tamaño máximo alzanzado por A
        # bucle principal ramificacion y poda (PODA IMPLICITA)
        while len(A)>0 and A[0][0] < self.fx:
            iterations += 1
            lenA = len(A)
            maxA = max(maxA, lenA)
            s_score, s = heapq.heappop(A)
            for child_score, child in self.branch(s_score, s):
                gen_states += 1
                if self.is_complete(child): # si es terminal
                    # es factible (pq branch solo genera factibles)
                    # falta ver si mejora la mejor solucion en curso
                    if child_score < self.fx:
                        self.fx, self.x = child_score, child
                else: # no es terminal
                    # lo metemos en el cjt de estados activos si
                    # supera la poda por cota optimista:
                    if child_score < self.fx:
                        heapq.heappush(A, (child_score, child) )
                    else:
                        podas_opt += 1
                        
        stats = { 'iterations':iterations,
                  'gen_states':gen_states,
                  'podas_opt':podas_opt,
                  'maxA':maxA}
        return self.fx, self.x, stats

def functionRyP(costMatrix):
    e = Ensamblaje(costMatrix)
    fx,x,stats = e.solve()
    return fx,x

######################################################################
#
#                        EXPERIMENTACIÓN
#
######################################################################

cjtAlgoritmos = {'naif': naive_solution,
                 'x_pieza': voraz_x_pieza,
                 'x_instante': voraz_x_instante,
                 'x_coste': voraz_x_coste,
                 'combina': voraz_combina,
                 'RyP': functionRyP}
cjtSolutionesIniciales = {
    'naif+RyP': naive_solution,
    'x_pieza+RyP': voraz_x_pieza,
    'x_instante+RyP': voraz_x_instante,
    'x_coste+RyP': voraz_x_coste,
    'combina+RyP': voraz_combina,
    'RyP': lambda x: (np.inf, None)
}

def probar_ejemplo():
    ejemplo = np.array([[7, 3, 7, 2],
                        [9, 9, 4, 1],
                        [9, 4, 8, 1],
                        [3, 4, 8, 4]], dtype=int)
    
    for label,function in cjtAlgoritmos.items():
        score,solution = function(ejemplo)
        print(f'Algoritmo {label:10}', solution, score)

def comparar_algoritmos(root_seed, low, high):
    print('talla',end=' ')
    for label in cjtAlgoritmos:
        print(f'{label:>10}',end=' ')
    print()
    numInstancias = 10
    for talla in range(5,15+1):
        dtalla = collections.defaultdict(float)

        np.random.seed(root_seed)
        seeds = np.random.randint(low=0, high=9999, size=numInstancias)

        for seed in seeds:
            cM = genera_instancia(talla, low=low, high=high, seed=seed)
            for label,function in cjtAlgoritmos.items():
                score,solution = function(cM)
                dtalla[label] += score

        print(f'{talla:>5}',end=' ')
        for label in cjtAlgoritmos:
            media = dtalla[label]/numInstancias
            print(f'{media:10.2f}', end=' ')
        print()

def comparar_sol_inicial(root_seed, low, high):
    
    numInstancias = 10
    stats_t = ['iterations',
                'gen_states',
                'podas_opt',
                'maxA']
    tallas = range(5,15+1)
    for stat in stats_t:
        print('-'*40 + f"   {stat}   " + '-'*40)
        print('talla',end=' ')
        for label in cjtAlgoritmos:
            print(f'{label:>10}',end=' ')
        print()
        for talla in range(5,15+1):
            dtalla = collections.defaultdict(float)

            np.random.seed(root_seed)
            seeds = np.random.randint(low=0, high=9999, size=numInstancias)

            for seed in seeds:
                cM = genera_instancia(talla, low=low, high=high, seed=seed)
                for label,function in cjtSolutionesIniciales.items():
                    score,solution = function(cM)
                    e = Ensamblaje(cM,solution)
                    fx,x,stats = e.solve()
                    dtalla[label] += stats[stat]
            print(f'{talla:>5}',end=' ')
            for label in cjtSolutionesIniciales:
                media = dtalla[label]/numInstancias
                print(f'{media:10.2f}', end=' ')
            print()

    
def probar_ryp():
    ejemplo = np.array([[7, 3, 7, 2],
                        [9, 9, 4, 1],
                        [9, 4, 8, 1],
                        [3, 4, 8, 4]], dtype=int)
    # scorevoraz, solvoraz = voraz_combina(ejemplo)
    # bb = Ensamblaje(ejemplo, solvoraz)
    bb = Ensamblaje(ejemplo)
    fx, x, stats = bb.solve()
    print(x,fx,compute_score(ejemplo,x))
    print(stats)

######################################################################
#
#                             PRUEBAS
#
######################################################################
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soluciones iniciales voraces al problema del ensamblaje por RyP.')
    
    parser.add_argument('-S', '--seed', dest='seed', action='store', type=int, default=42, 
                        help='Semilla para la generación de instancias.')
    
    parser.add_argument('-L', '--low', dest='low', action='store', type=int, default=1, 
                        help='Valor inferior para generación de instancias.')
    
    parser.add_argument('-H', '--high', dest='high', action='store', type=int, default=4, 
                        help='Valor superior (no incluído) para generación de instancias.')

    parser.add_argument('-R', '--ryp', dest='ryp', action='store_true', default=False, 
                        help='Prueba RyP.')

    parser.add_argument('-A', '--algoritmos', dest='algoritmos', action='store_true', default=False, 
                        help='Comparar algoritmos voraces con RyP.')
    
    parser.add_argument('-I', '--inicial', dest='inicial', action='store_true', default=False, 
                        help='Comparar soluciones iniciales voraces en RyP.')

    args = parser.parse_args()

    SEED = args.seed
    LOW  = args.low
    HIGH = args.high

    if args.ryp:
        probar_ryp()
        
    if args.algoritmos:
        comparar_algoritmos(root_seed=SEED, low=LOW, high=HIGH)
        
    if args.inicial:
        comparar_sol_inicial(root_seed=SEED, low=LOW, high=HIGH)

