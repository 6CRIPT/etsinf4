######################################################################
# Práctica realizada por:
# - César Martínez Chico
# - JAIME CANDEL MARTINEZ
######################################################################

import time
import math
import numpy as np
import collections

class DiGraph:
    '''
    Grafo dirigido representado mediante listas de adyacencia
    y de incidencia.
    
    Hay 2 atributos. Si tenemos aristas de u a v con peso w

    - self.forward es un diccionario que a cada vértice u le asocia
      una lista de pares (v,w)

    - self.backwards es un diccionario que a cada vértice v le asocia
      una lista de pares (u,w)

    forward es una lista de adyacencia (se vió en EDA)

    backwards es como la lista de adyacencia del grafo transpuesto
    (darle la vuelta a la orientación de las aristas)
    '''
    
    def __init__(self):
        self.forward = {} # aristas que salen de cada vértice
        # backwards solo se utiliza por Dijkstra cuando reverse=True
        # pero hay una cota que lo usa muchas veces
        self.backwards = {} # aristas que llegan a cada vértice

    def nV(self):
        '''
        devuelve |V| nº vértices
        '''
        return len(self.forward)

    def add_node(self, u):
        '''
        añade un nuevo nodo
        en este caso no pasa nada si ya estaba
        '''
        if u not in self.forward:
            self.forward[u] = []
            self.backwards[u] = []

    def add_edge(self, u, v, weight=1):
        '''
        añade una arista de u a v con peso weight
        asume que esa arista no estaba previamente
        '''        
        # we assume there is no edge (u,v)
        self.add_node(u) # just in case
        self.add_node(v) # just in case
        self.forward[u].append((v,weight))
        self.backwards[v].append((u,weight))

    def nodes(self):
        '''
        para poder iterar sobre los nodos desde fuera
        de la clase sin acceder explícitamente a self.forward
        '''
        return self.forward.keys()
    
    def edges_from(self, u):
        '''
        para poder iterar sobre las aristas que salen de u
        sin acceder explícitamente a un atributo
        '''
        for v,w in self.forward[u]:
            yield v,w
    
    def is_edge(self, u, v):
        '''
        se limita a decir si hay una arista de u a v
        '''
        return any(dst == v for (dst,w) in self.forward[u])

    def weight_edge(self, u, v):
        '''
        devulve peso de la arista de u a v
        returns None if there is no such edge
        '''
        for (dst,w) in self.forward[u]:
            if v == dst:
                return w

    def lowest_out_weight(self, vertex, forbidden=()):
        '''
        devuelve el peso de la arista de menor peso
        de las que salen de vertex pero no llegan
        a ninguna de forbidden
        '''
        return min((w for v,w in self.forward[vertex]
                    if v not in forbidden),default=float('inf'))

    def path_weight(self, path):
        '''
        Devuelve el coste del camino path si este existe
        En caso contrario devuelve None
        '''
        score = 0
        for u,v in zip(path,path[1:]):
            w = self.weight_edge(u,v)
            if w is None:
                return None
            score += w
        return score

    def check_TSP(self, cycle):
        '''
        check if cycle is a TSP of the graph
        return bool,weight_of_cycle
        '''
        setcycle = set(cycle[:-1])
        if (cycle[0] != cycle[-1] or
            len(setcycle) != len(cycle)-1 or
            setcycle != set(self.forward)):
            return False,0
        # follow the path:
        score = self.path_weight(cycle)
        if score is None:
            return False,0
        return True, score

    def Dijkstra(self, src, reverse=False):
        '''
        Lanza el algoritmo de Dijkstra en el grafo
        desde el vértice src
        Si ponemos reverse=True calcula las distancias
        desde cada vértice hasta src y no desde src al resto
        '''
        dist = { src:0 } # asocia una distancia a cada vértice
        pred = {} # para recuperar el camino de menor peso
        fixed = set() # cjt de vértices que ya han sido fijados
        eligible = { src } # cjt con src como elemento inicial
        adj = self.backwards if reverse else self.forward
        while len(eligible)>0:
            d,u = min( (dist[v],v) for v in eligible)
            eligible.remove(u) # lo sacamos de eligible
            fixed.add(u)       # y lo añadimos a fixed
            # recorremos backwards en lugar de forward:
            for v,weight in adj[u]: # actualizamos cota de adyacentes a v
                if v not in fixed:
                    if v not in dist:
                        pred[v], dist[v] = u, dist[u] + weight
                        eligible.add(v)
                    elif dist[v] > dist[u] + weight:
                        pred[v], dist[v] = u, dist[u] + weight
        return dist, pred

    def Dijkstra1dst(self, src, dst, avoid=()):
        '''
        Lanza el algoritmo de Dijkstra en el grafo
        desde el vértice src hasta el vértice dst
        y sólo devuelve la distancia.
        Evita utilizar los vértices en avoid
        '''
        dist = { src:0 } # asocia una distancia a cada vértice
        fixed = set() # cjt de vértices que ya han sido fijados
        eligible = { src } # cjt con src como elemento inicial
        while len(eligible)>0:
            d,u = min( (dist[v],v) for v in eligible)
            if u == dst:
                break
            eligible.remove(u) # lo sacamos de eligible
            fixed.add(u)       # y lo añadimos a fixed
            # recorremos backwards en lugar de forward:
            for v,weight in self.forward[u]: # actualizamos cota de adyacentes a v
                if v not in fixed and v not in avoid:
                    if v not in dist:
                        dist[v] = dist[u] + weight
                        eligible.add(v)
                    elif dist[v] > dist[u] + weight:
                        dist[v] = dist[u] + weight
        return dist.get(dst, float('inf'))
    
    def StronglyCC(self):
        '''
        Algoritmo de Tarjan para calcular las componentes
        fuertemente conexas del grafo
        https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        adaptado de
        https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/
        '''
        sccs = []
        disc = {}
        low = {}
        time = [0]
        st = [] # stack
        stset = set()

        def SCCUtil(u): # función (closure) auxiliar, no es un método
            disc[u] = time[0]
            low[u] = time[0]
            time[0] += 1
            st.append(u)
            stset.add(u)
            
            for v,w in self.forward[u]:
                if v not in disc:
                    SCCUtil(v)
                    low[u] = min(low[u], low[v])
                elif v in stset:
                    low[u] = min(low[u], disc[v])

            w = None
            if low[u] == disc[u]:
                scc = []
                while w != u:
                    w = st.pop()
                    scc.append(w)
                    stset.discard(w)
                sccs.append(scc)
            
        for v in self.forward:
            if v not in disc:
                SCCUtil(v)
                
        return sccs

######################################################################
#
#                     GENERACIÓN DE INSTANCIAS
#
######################################################################

def generate_random_digraph(nV, dimacsList=None, seed=None):
    '''
    recibe nº vértices
    tofile puede ser una lista Python (se la damos vacía),
    en cuyo caso escribe en ella el grafo en formato Dimacs:
    http://prolland.free.fr/works/research/dsat/dimacs.html

    seed permite inicializar la generación de valores
    pseudo-aleatorios para facilitar la reproducibilidad
    de los experimentos
    '''
    rng = np.random.default_rng(seed)

    # calculamos unas coordenadas para cada vértice en un plano 2D
    xCoord = rng.uniform(0, 1000, size=nV)
    yCoord = rng.uniform(0, 1000, size=nV)

    # calculamos los grados de salida de todos vértices:
    gradomin = min(3,nV-1)
    gradomax = min(8,nV-1)

    outdegrees = rng.integers(gradomin, gradomax, size=nV)

    nE = sum(outdegrees)
    # el grafo que vamos a devolver:
    g = DiGraph()
    if dimacsList is not None:
        dimacsList.append(f'FORMAT {nV} {nE}')
    for vertex in range(nV):
        destinations = set(range(nV))
        destinations.remove(vertex)
        destinations = np.array(list(destinations))

        rng.shuffle(destinations)
        destinations = destinations[:outdegrees[vertex]]
        
        for destination in destinations:
            euclidean = math.hypot(xCoord[vertex]-xCoord[destination],
                                   yCoord[vertex]-yCoord[destination])
            # no es la distancia euclídea, aunque está relacionada con
            # ésta no es una métrica...
            distance = int(100*rng.uniform(1,1.5) * euclidean)
            if dimacsList is not None:
                dimacsList.append(f'e {vertex} {destination} {distance}')
            g.add_edge(vertex, destination, distance)
    return g


def generate_random_digraph_1scc(nV, dimacsList=None, seed=None):
        nsccs = 2  # para entrar en el while la 1a vez
        intentos = 0
        while nsccs>1:
            intentos += 1
            if intentos > 100:
                raise(Exception("too much attempts generating graph"))
            if dimacsList is not None:
                dimacsList.clear()
            g = generate_random_digraph(nV,dimacsList,seed)
            if seed is not None:
                seed += 1
            nsccs = len(g.StronglyCC())
        return g

def load_dimacs_graph(dimacsString, symmetric=False):
    g = DiGraph()
    for line in dimacsString.split('\n'):
        if line.startswith('FORMAT'):
            nV,nE = map(int,line.split()[1:])
            for v in range(nV):
                g.add_node(v)
        elif line.startswith('e '):
            u,v,w = map(int,line.split()[1:])
            g.add_edge(u,v,w)
            if symmetric:
                g.add_edge(v,u,w)
    return g

######################################################################
#
#                       RAMIFICACIÓN Y PODA
#
######################################################################

import heapq

class BranchBoundImplicit:
    def solve(self):
        startTime = time.perf_counter()
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
                    if self.is_factible(child):
                        # falta ver si mejora la mejor solucion en curso
                        child_score, child = self.get_factible_state(child_score,
                                                                     child)
                        if child_score < self.fx:
                            self.fx, self.x = child_score, child
                else: # no es terminal
                    # lo metemos en el cjt de estados activos si
                    # supera la poda por cota optimista:
                    if child_score < self.fx:
                        heapq.heappush(A, (child_score, child) )
                    else:
                        podas_opt += 1
                        
        endTime = time.perf_counter()                        
        stats = { 'time': endTime-startTime,
                  'iterations':iterations,
                  'gen_states':gen_states,
                  'podas_opt':podas_opt,
                  'maxA':maxA }
        
        return self.fx, self.x, stats

class BranchBoundExplicit:
    def solve(self):
        startTime = time.perf_counter()
        A = [ self.initial_solution() ] # cola de prioridad
        iterations = 0 # nº iteraciones
        gen_states = 0 # nº estados generados
        podas_opt  = 0 # nº podas por cota optimista
        maxA       = 0 # tamaño máximo alzanzado por A
        cost_expl  = 0 # coste poda explícita
        podas_expl = 0 # podas dentro de A con la poda explícita
        # bucle principal ramificacion y poda (PODA IMPLICITA)
        while len(A)>0:
            iterations += 1
            lenA = len(A)
            maxA = max(maxA, lenA)
            s_score, s = heapq.heappop(A)
            for child_score, child in self.branch(s_score, s):
                gen_states += 1
                if self.is_complete(child): # si es terminal
                    if self.is_factible(child):
                        # falta ver si mejora la mejor solucion en curso
                        child_score, child = self.get_factible_state(child_score,
                                                                     child)
                        if child_score < self.fx:
                            self.fx, self.x = child_score, child
                            # poda explícita
                            cost_expl += len(A)
                            lenAprev = len(A)
                            A = [ (score,st) for (score,st) in A
                                  if score < child_score]
                            podas_expl += lenAprev - len(A)
                            # this method was called 'builheap' in EDA:
                            heapq.heapify(A) # transform into a heap, 
                else: # no es terminal
                    # lo metemos en el cjt de estados activos si
                    # supera la poda por cota optimista:
                    if child_score < self.fx:
                        heapq.heappush(A, (child_score, child) )
                    else:
                        podas_opt += 1
                        
        endTime = time.perf_counter()                        
        stats = { 'time' : endTime-startTime,
                  'iterations' : iterations,
                  'gen_states' : gen_states,
                  'podas_opt' : podas_opt,
                  'maxA' : maxA,
                  'podas_expl' : podas_expl,
                  'cost_expl' : cost_expl }
        
        return self.fx, self.x, stats

class TSP:

    def __init__(self, graph, first_vertex=0):
        self.G = graph # DiGraph
        self.first_vertex = first_vertex
        self.fx = float('inf')
        self.x = None
        
    def is_complete(self, s):
        '''
        s es una solución parcial
        '''
        return len(s) == self.G.nV()

    def is_factible(self, s):
        '''
        asumimos s is complete, falta ver si hay una arista desde
        el último vértice s[-1] hasta el primero s[0]
        '''
        return self.G.is_edge(s[-1],s[0])

    def get_factible_state(self, s_score, s):
        '''
        En el caso del TSP se encarga de añadir la arista que
        vuelve al origen Dependiendo del tipo de cota optimista,
        hay que corregir el valor de la parte conocida y desconocida,
        en el caso general no nos la jugamos (mejor optimizar para
        casos particulares):
        '''
        s_new = s+[s[0]]
        return self.G.path_weight(s_new), s_new

class TSP_Cota1(TSP):
    '''
    Versión con cota trivial. Solamente se suma la parte conocida.
    La parte desconocida vale 0.
    '''

    def initial_solution(self):
        initial = [ self.first_vertex ]
        initial_score = 0
        return (initial_score, initial)

    def branch(self, s_score, s):
        '''
        s_score es el score de s
        s es una solución parcial
        '''
        lastvertex = s[-1]
        for v,w in self.G.edges_from(lastvertex):
            if v not in s:
                yield (s_score + w, s+[v])

class TSP_Cota4(TSP):
    '''
    Las aristas de mínimo coste que parten de cada vértice
    no visitado (más el último visitado). 
    Es fácil calcularla de manera incremental.
    '''
    def initial_solution(self):
        self.menores_pesos = {} #CON EL SELF AQUI LO DEFINES PARA TODA LA CLASE. CLAVE DE PYTHON
        initial = [ self.first_vertex ]
        initial_score = 0
        for v in self.G.nodes():
            self.menores_pesos[v] = self.G.lowest_out_weight(v)
        return (sum(self.menores_pesos.values()), initial) #sumamos todos los minimos pesos
    
    def branch(self, s_score, s): #RECUERDA QUE SCORE, ES LA COTA, O SEA PARTE CONOCIDA + DESCONOCIDA. 
        lastvertex = s[-1]
        aristas = self.G.edges_from(lastvertex)
        #minima_arista = min(aristas, key = lambda i: i[1])#recuerda, ordenamos la lista aristas, pero ordenamos sus elementos tras aplicar la funcion key
        # que se APLICA A TODOS LOS ELEMENTOS DE ARISTAS --> es decir que i es cada tupla arista, y simplemente devolvemos su peso con i[1]
        aristas = sorted(aristas, key = lambda i: i[1])
        for v,w in aristas:
            if v not in s:
                yield (s_score + w - self.menores_pesos[lastvertex], s+[v]) #le restas el ultimo porque das por hecho con la cota que habias metido el optimo, por tanto si has metido el optimo no sumas nada (ya estaba sumado) si no, pagas la diferencia.

class TSP_Cota5(TSP):
    '''
    Las aristas de mínimo coste que parten de cada vértice
    no visitado (más el último visitado)
    y llegan a vértices no visitados o al vértice origen.
    No es incremental.
    '''
    def initial_solution(self):
        initial = [ self.first_vertex ]
        initial_score = 0
        for v in self.G.nodes():
            initial_score += self.G.lowest_out_weight(v)
        return (initial_score, initial)
    
    def cota5(self, child):
        #cota = 0
        #for vertice in [ vertice for vertice in self.G.nodes() if vertice not in child]: #solo iteramos los vertices que nos interesan
            #cota += min([arista for arista in self.edges_from(vertice) if arista[0] not in sol], key = lambda x: x[1])[1] #basicamente ordenamos por peso, pero solo si el vertice de destino no esta en la solucion y nos quedamos con la minima arista y sumamos su coste   
            #cota += self.G.lowest_out_weight(vertice, avoid = set(child[1:]))
        return sum(self.G.lowest_out_weight(vertice, forbidden = set(child[1:])) for vertice in self.G.nodes() if vertice not in child[:-1]) 
        #sumas todas las aristas de minimo peso que salgan de los vertices que no esten la solucion, y que no lleguen a ninguno que este en el hijo, menos el primero porque podriamos cerrar el ciclo 
        #return cota      
        
    
    def branch(self, sol_score, sol):
        for arista in self.G.edges_from(sol[-1]): #aristas en el ultimo vertice
            child = sol
            if arista[0] not in child:
                #child.append(arista[0]) #metemos el vertice 
                child_score = self.G.path_weight(child + [arista[0]]) + self.cota5(child + [arista[0]]) #sumamos su coste + LA COTA 
                yield (child_score, child + [arista[0]]) #devolvemos
                
class TSP_Cota6(TSP):
    '''
    Completar el ciclo utilizando un camino.
    También se puede calcular de modo incremental tras
    aplicar el algoritmo de Dijkstra desde el vértice
    inicial del camino sobre el grafo traspuesto
    (usando las aristas en dirección contraria) para así
    calcular en una sola pasada los caminos desde cada
    vértice al inicial.
    '''
    
    def initial_solution(self):
        self.caminos_minimos = {} #vamos a guardar lo que cuesta llegar desde cada vertice hasta el inicial, solo nos interesa key = vertice y el valor es el valor minimo para llegar.
        initial_sol = [self.first_vertex]
        
        self.caminos_minimos, _ =  self.G.Dijkstra(initial_sol[-1], reverse = True) #reversed porque te lo dice arriba que sea con las aristas en contrario, los guardamos en el diccionario
            #siempre que llamemos a caminos minimos hay que usar self.
        #QUITAMOS EL FOR PUES DIJKSTRA DIRECTAMENTE PARA EL VERTICE SOLUCION TE CREA EL DICCIONARIO.
        return (self.caminos_minimos[self.first_vertex], initial_sol) # es un 0 el score inicial, ya que para llegar del vertice 0 (first_vertex) a sí mismo, el coste es 0 ya que estamos ahi jaja
    def branch(self, s_score, child):    
        lastvertex = child[-1]
        for v,w in self.G.edges_from(lastvertex):
            if v not in child:
                #importante SI HACES APPEND, HAZ LUEGO POP, PERO MEJOR NO HACERLO, PASALO DIRECTAMENTE EN EL YIELD.
                yield (s_score + w - self.caminos_minimos[lastvertex] + self.caminos_minimos[v], child + [v]) #le restas el ultimo porque das por hecho con la cota que habias metido el optimo, por tanto si has metido el optimo no sumas nada (ya estaba sumado) si no, pagas la diferencia.
                #primero restas LA COTA DE lastvertex, PORQUE SI NO, ESTARIAS ACUMULANDO COTAS.

class TSP_Cota7(TSP):
    '''
    Completar el ciclo utilizando un camino
    pero, a diferencia de la cota 6, ese camino
    no puede pasar por vertices visitados (salvo
    el primero y el último de la solución parcial).
    No admite sol. incremental.
    '''
    
    def initial_solution(self):
        initial = [self.first_vertex]
        initial_score = self.G.Dijkstra1dst(initial[-1], initial[-1])
        return (initial_score, initial)
    def branch(self, s_score, s):
        child = s # el camino hijo de momento igual que el que ramificamos.
        for arista in self.G.edges_from(child[-1]):
            if arista[0] not in child: #si el vertice no esta en la sol
                child_score = self.G.path_weight(child + [arista[0]]) + self.G.Dijkstra1dst(arista[0], self.first_vertex, avoid = child[1:])
                yield(child_score, child + [arista[0]])

                



######################################################################
#
#                        CLASES DERIVADAS
#
######################################################################

class TSP_Cota1I(TSP_Cota1, BranchBoundImplicit):
    pass

class TSP_Cota1E(TSP_Cota1, BranchBoundExplicit):
    pass

class TSP_Cota4I(TSP_Cota4, BranchBoundImplicit):
    pass

class TSP_Cota4E(TSP_Cota4, BranchBoundExplicit):
    pass

class TSP_Cota5I(TSP_Cota5, BranchBoundImplicit):
    pass

class TSP_Cota5E(TSP_Cota5, BranchBoundExplicit):
    pass

class TSP_Cota6I(TSP_Cota6, BranchBoundImplicit):
    pass

class TSP_Cota6E(TSP_Cota6, BranchBoundExplicit):
    pass

class TSP_Cota7I(TSP_Cota7, BranchBoundImplicit):
    pass

class TSP_Cota7E(TSP_Cota7, BranchBoundExplicit):
    pass

# ir descomentando a medida que se implementen las cotas
repertorio_cotas = [('Cota1I',TSP_Cota1I),
                    ('Cota1E',TSP_Cota1E),
                    ('Cota4I',TSP_Cota4I),
                    ('Cota4E',TSP_Cota4E),
                    ('Cota5I',TSP_Cota5I),
                    ('Cota5E',TSP_Cota5E),
                    ('Cota6I',TSP_Cota6I),
                    ('Cota6E',TSP_Cota6E),
                    ('Cota7I',TSP_Cota7I),
                    ('Cota7E',TSP_Cota7E)
                    ]

######################################################################
#
#                        EXPERIMENTACIÓN
#
######################################################################

# algunas cotas son tan lentas que mejor fijarles una talla máxima
tallaMax = {'Cota1I':15,
            'Cota1E':15,
            'Cota4I':20,
            'Cota4E':20,
            'Cota5I':20,
            'Cota5E':20,
            'Cota6I':15,
            'Cota6E':15,
            'Cota7I':15,
            'Cota7E':15}

def prueba_generador():
    for nV in range(5,1+max(tallaMax.values())):
        print("-"*80)
        print(f"Probando grafos de talla {nV}")
        g = generate_random_digraph_1scc(nV)
        for nombre,clase in repertorio_cotas:
            if nV <= tallaMax[nombre]:
                print(f'\n  checking {nombre}')
                tspi = clase(g)
                fx,x,stats = tspi.solve()
                print(' ',fx,x,stats)
                if x is not None:
                    print(' ',g.check_TSP(x))
    

# estas semillas se han probado para comprobar que los grafo generados
# tienen una sola componente fuertemente conexa
seeds = {                    
    10 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    11 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    12 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    13 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    14 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    15 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    16 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21],
    17 : [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    18 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    19 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    20 : [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
}
                    
def experimento():
    '''
    - Probar con varias tallas entre 10 y 20
    - Para cada talla generar entre 5 y 20 instancias (según lo lento
      que funcione)
    - IMPORTANTE probar diversos algoritmos con LAS MISMAS instancias
    - Descartar las que no den solución (las que sacan None,float('inf'))
    - Mostrar valores medios de todas las instancias de cada talla.
    - Sacar los datos de manera que sea fácil realizar una interpretacion
    '''

    stats_keys = ['time', 'iterations', 'gen_states', 'podas_opt', 'maxA' ] 
    print(f'Initializing experiment...\n')
    all_sizes = range(10, 20+1)
    all_stats = {}
    num_instances = 10
    for nV in range(10,1+max(tallaMax.values())):
        print(f'  Checking graph size {nV}  '.center(50, '#'))
        size_dict = all_stats.get(nV, {})
        for instance in range(num_instances):
            print(f'\tInstance num {instance}')
            g = generate_random_digraph_1scc(nV) # porque fuertemente conexo
            for nombre,clase in repertorio_cotas:
                print(f'\t\tChecking {nombre}')
                cota_list = size_dict.get(nombre, [])
                tspi = clase(g)
                fx,x,stats = tspi.solve()
                cota_list.append(stats)
                size_dict[nombre] = cota_list
        for nombre,clase in repertorio_cotas:
            cota_list = size_dict.get(nombre, [])
            mean_dict = {}
            for key in cota_list[0].keys():
                mean_dict[key] = sum(d[key] for d in cota_list) / len(cota_list)
            size_dict[nombre] = mean_dict
        all_stats[nV]=size_dict
    for key in stats_keys:
        print(key.center(20,' ').center(100, '-'))
        print('talla',end=' ')
        for nombre,clase in repertorio_cotas:
            print(f'{nombre:>10}',end=' ')
        print()
        for talla in range(10,1+max(tallaMax.values())):
            print(f'{talla:>5}', end=' ')
            for nombre,clase in repertorio_cotas:
                print(f'{all_stats[talla][nombre][key]:10.2f}', end=' ')
            print()



######################################################################
#
#                            PRUEBA
#
######################################################################

ejemplo_teoria = '''
FORMAT 10 18
e 0 1  1
e 0 2 15
e 0 3  2
e 1 3  3
e 1 4 13
e 2 3 11
e 2 5  4
e 3 4  5
e 3 5  8
e 3 6 12
e 4 7  9
e 5 6 16
e 5 8 10
e 6 7 17
e 6 8  1
e 6 9  6
e 7 9 14
e 8 9  7
'''

def prueba_mini():
    g = load_dimacs_graph(ejemplo_teoria,True)
    for nombre,clase in repertorio_cotas:
        print(f'----- checking {nombre} -----')
        tspi = clase(g)
        fx,x,stats = tspi.solve()
        print(fx,x,stats)

######################################################################
#
#                            MAIN
#
######################################################################
            
if __name__ == '__main__':
    #prueba_mini()
    # prueba_generador()
    experimento()

