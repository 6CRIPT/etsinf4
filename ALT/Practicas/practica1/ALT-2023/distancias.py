import numpy as np

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int64)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y, threshold=None):
    # a partir de la versión levenshtein_matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int64)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    res = []
    i,j = lenX, lenY
    while i > 0 or j > 0:         
        if D[i,j-1] + 1 == D[i, j]:
            # insercion
            res.append(('',y[j - 1]))
            j = j -1
        elif D[i-1, j] + 1 == D[i, j]:
            # borrado
            res.append((x[i - 1], ''))
            i = i - 1
        else:
            # son distintos en diagonal
            res.append((x[i-1], y[j-1]))
            i = i - 1
            j = j - 1
    res.reverse()
    return D[lenX, lenY], res # COMPLETAR Y REEMPLAZAR ESTA PARTE

def levenshtein_reduccion(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    
    # Crear dos vectores numpy de dimensiones (lenX + 1, 1) y (lenY + 1, 1)
    # como dice el enunciado para eliminar la matriz
    filaPrevia = np.arange(lenY + 1).reshape(-1, 1)
    filaActual = np.zeros((lenY + 1, 1), dtype=np.int64)
    
    for i in range(1, lenX + 1):
        filaActual[0] = i  
        
        for j in range(1, lenY + 1):
            coste = int(x[i - 1] != y[j - 1])
            filaActual[j] = min(
                filaActual[j - 1] + 1,        # Eliminación
                filaPrevia[j] + 1,              # Inserción
                filaPrevia[j - 1] + coste        # Sustitución
            )
        
        # actualizamos las filas
        filaPrevia, filaActual = filaActual, filaPrevia
    
    
    return filaPrevia[lenY][0]
    #REDUCCION ESPACIAL HECHO: CÉSAR

def levenshtein(x, y, threshold):
    # completar versión con reducción coste espacial Y UMBRAL
    lenX, lenY = len(x), len(y)
    
    if threshold == None:
        threshold = lenX
    # Crear dos vectores numpy de dimensiones (lenY + 1, 1) y (lenY + 1, 1)
    # como dice el enunciado para eliminar la matriz
    filaPrevia = np.arange(lenY + 1).reshape(-1, 1)
    filaActual = np.zeros((lenY + 1, 1), dtype=np.int64)
    
    for i in range(1, lenX + 1):
        filaActual[0] = i  
        
        for j in range(1, lenY + 1):
            coste = int(x[i - 1] != y[j - 1])
            filaActual[j] = min(
                filaActual[j - 1] + 1,        # Eliminación
                filaPrevia[j] + 1,              # Inserción
                filaPrevia[j - 1] + coste        # Sustitución
            )
        
        # actualizamos las filas
        filaPrevia, filaActual = filaActual, filaPrevia

        if filaPrevia.min() > threshold:
            return threshold + 1
    
    return filaPrevia[lenY][0]
    #REDUCCION ESPCIAL + UMBRAL HECHO: CÉSAR NO BRO TE PASAS YIYI
    

def levenshtein_cota_optimista(x, y, threshold):
    return 0 # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein restringida con matriz
    lenX, lenY = len(x), len(y)
    # COMPLETAR
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int64)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + 1 if x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] else np.inf
            )
    return D[lenX, lenY]

def damerau_restricted_edicion(x, y, threshold=None):
    # partiendo de damerau_restricted_matriz añadir recuperar
    # secuencia de operaciones de edición
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int64)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + 1 if x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] and i > 1 and j > 1 else np.inf
            )
    res = []
    i,j = lenX, lenY
    while i > 0 or j > 0:         
        if D[i,j-1] + 1 == D[i, j]:
            # Insercion
            res.append(('',y[j - 1]))
            j = j -1
        elif D[i-1, j] + 1 == D[i, j]:
            # Borrado
            res.append((x[i - 1], ''))
            i = i - 1
        elif x[i-1] == y[j-1] and D[i - 1, j - 1] == D[i, j]:
            # Son distintos en diagonal
            res.append((x[i-1], y[j-1]))
            i = i - 1
            j = j - 1
        elif x[i-1] != y[j-1] and D[i - 1, j - 1] + 1 == D[i, j]:
            # Son distintos en diagonal
            res.append((x[i-1], y[j-1]))
            i = i - 1
            j = j - 1
        else:
            # Intercambio de dos adyacentes
            res.append((x[i-2]+x[i-1],y[j-2]+y[j-1]))
            i = i - 2
            j = j - 2
            
    res.reverse()   
    return D[lenX, lenY], res # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)
    
    if threshold == None:
        threshold = lenX

    filaPrevia = np.arange(lenY + 1)
    filaActual = np.zeros(lenY + 1, dtype=np.int64)
    
    filaActual[0] = 1

    for j in range(1,lenY + 1):
        coste = int(x[0] != y[j - 1])
        filaActual[j] = min(
            filaActual[j - 1] + 1,        # Eliminación
            filaPrevia[j] + 1,              # Inserción
            filaPrevia[j - 1] + coste        # Sustitución
        )
    filaPreviaPrevia = np.copy(filaPrevia)
    filaPrevia, filaActual = filaActual, filaPrevia
    if filaPrevia.min() > threshold:
        return threshold + 1
    for i in range(2, lenX + 1):
        filaActual[0] = i  
        
        for j in range(1, lenY + 1):
            coste = int(x[i - 1] != y[j - 1])

            filaActual[j] = min(
                filaActual[j - 1] + 1,        # Eliminación
                filaPrevia[j] + 1,              # Inserción
                filaPrevia[j - 1] + coste,        # Sustitución
                filaPreviaPrevia[j - 2] + 1 if x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] else np.inf
            )
        filaPreviaPrevia = np.copy(filaPrevia)

        # actualizamos las filas
        filaPrevia, filaActual = filaActual, filaPrevia
        if filaPrevia.min() > threshold:
            return threshold + 1
    
    return filaPrevia[lenY]

def damerau_intermediate_matriz(x, y, threshold=None):
    lenX, lenY = len(x), len(y)

    if threshold is None:
        threshold = lenX

    prev_row = np.arange(lenY + 1)
    current_row = np.zeros(lenY + 1, dtype=np.int64)

    current_row[0] = 1

    for j in range(1, lenY + 1):
        cost = int(x[0] != y[j - 1])
        current_row[j] = min(
            current_row[j - 1] + 1,  # Eliminación
            prev_row[j] + 1,        # Inserción
            prev_row[j - 1] + cost  # Sustitución
        )

    prev_prev_row = np.copy(prev_row)
    prev_row, current_row = current_row, prev_row

    if prev_row.min() > threshold:
        return threshold + 1

    for i in range(2, lenX + 1):
        current_row[0] = i

        for j in range(1, lenY + 1):
            cost = int(x[i - 1] != y[j - 1])

            transposition_cost = np.inf
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                transposition_cost = prev_prev_row[j - 2] + 1

            current_row[j] = min(
                current_row[j - 1] + 1,    # Eliminación
                prev_row[j] + 1,          # Inserción
                prev_row[j - 1] + cost,   # Sustitución
                transposition_cost        # Transposición
            )

        prev_prev_row = np.copy(prev_row)
        prev_row, current_row = current_row, prev_row

        if prev_row.min() > threshold:
            return threshold + 1

    return prev_row[lenY]


def damerau_intermediate_edicion(x, y, threshold=None):
    # partiendo de matrix_intermediate_damerau añadir recuperar
    # secuencia de operaciones de edición
    # completar versión Damerau-Levenstein intermedia con matriz
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE
    
def damerau_intermediate(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    return min(0,threshold+1) # COMPLETAR Y REEMPLAZAR ESTA PARTE

opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_r':     damerau_restricted,
    'damerau_im':    damerau_intermediate_matriz,
    'damerau_i':     damerau_intermediate
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}
