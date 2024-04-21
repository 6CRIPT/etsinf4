def farolas(x,f,r):
    resul = []
    derecha = x[0]-1 #inicial, para primera iteracion del bucle cogemos el primero, pues le quitamos uno
    ultima = -1 #ultima farola usada, -1 para la 1 del bucle
    for punto in x:
        if punto > derecha:
            derecha, ultima = max((f[i] + r[i],i) for i in range(ultima + 1, len(f)) if f[i] - r[i] <= punto)
            """
            la idea es quedarte con la farola que mas radio tenga por la derecha
            no se incluyen las que tiene ese punto a la izda
            de tal forma que minimo cubra ese punto (por la izquierda)
            entonces para cubrir el primer punto coges la farola (por la derecha)
            que mas radio tiene, compruebas que lo cubra a ese punto por la izquierda
            y actualizas valores; para el siguiente punto, no te renta farolas que 
            tenga a su izquierda, ya que estaran cubiertas por la farola anterior
            asi que solo te fijas en las de la derecha.
            """
            resul.append(ultima)
            if ultima == f[-1]: 
                break
            """
            si ya hemos metido la ultima farola, los demas puntos que pueda haber a la derecha
            deberán estar cubiertos, por tanto ahorramos iteracciones de puntos finales.
            """                       
    return resul
x = [3, 4, 4, 7, 10]
f = [1,2,3,3,7,8]
r = [1,4,3,2,1,2]
print(farolas(x,f,r))
