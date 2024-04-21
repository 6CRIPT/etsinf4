import numpy as np

def levenshtein_distance(x, y):
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

# Ejemplo de uso:
word1 = "kitten"
word2 = "sitting"
distance = levenshtein_distance(word1, word2)
print(f"Distancia de Levenshtein entre '{word1}' y '{word2}': {distance}")
