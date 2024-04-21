import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
from spellsuggester import SpellSuggester
from distancias import *
class SAR_Indexer: 
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [
        ("all", True), ("title", True), ("summary", True), ("section-name", True), ('url', False),
    ]
    def_field = 'all'
    PAR_MARK = '%'
    
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'sindex', 'ptindex', 'docs', 'weight', 'articles',
                  'tokenizer', 'stemmer', 'show_all', 'use_stemming', 'positional']

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.urls = set()   # hash para las urls procesadas,
        self.urlsList = list()
        self.index = {}     # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.sindex = {}    # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {}   # hash para el indice permuterm.
        self.docs = {}      # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.numDocs = 0
        self.weight = {}    # hash de terminos para el pesado, ranking de resultados.
        self.articles = {}  # hash de articulos --> clave: entero(artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.numArt = 0
        self.tokenizer = re.compile("\W+")          # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish')   # stemmer en castellano
        self.show_all = False     # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()


        """ Incorporación de los atributos nuevos utilizados en ALT """
        self.use_spelling = False
        self.speller      = None

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v:bool):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v

    def set_spelling(self, use_spelling:bool, distance:str=None,
                threshold:int=None):
        """
            self.use_spelling a True activa la corrección ortográfica
            EN LAS PALABRAS NO ENCONTRADAS, en caso contrario NO utilizará
            corrección ortográfica
            input: "use_spell" booleano, determina el uso del corrector.
            "distance" cadena, nombre de la función de distancia.
            "threshold" entero, umbral del corrector
        """
        self.use_spelling = use_spelling
        if self.use_spelling:
            self.speller = SpellSuggester(opcionesSpell, list(self.index['all'].keys()), distance, threshold)

    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):

        """
        Guarda la información del índice en un fichero en formato binario
        """
        
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario
        
        """
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario

        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """
        
        Recorre recursivamente el directorio o fichero "root" 
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        file_or_dir = Path(root)
        
        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in files:
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos 
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article
                
    
    def index_file(self, filename:str):
        """
        Indexa el contenido de un fichero.
        
        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado

        """

        # Asignar al documento su docid
        self.numDocs += 1
        self.docs[self.numDocs] = filename

        for i, line in enumerate(open(filename)):
            j = self.parse_article(line)
            # Comprobar si el articulo ha ya ha sido indexado 
            if self.already_in_index(j):
                continue

            # Si no, se añade a la lista de indexados
            self.urls.add(j['url'])
            # Se asigna al artículo su artid
            self.numArt += 1
            self.articles[self.numArt] = [self.numDocs, i]
            #Se extrae el texto del artículo

            if self.multifield:
                for field, tokenizar in self.fields:
                    text = j[field]
                    if tokenizar:
                        terms = self.tokenize(text)
                    else:
                        terms = text
                    # Insertar los terminos en el diccionario
                    if field == 'url':
                        self.ins_dic(field, terms, self.numArt)
                    else:
                        for indice, term in enumerate(terms):
                            if self.positional:
                                self.ins_dic_positional(field, term, self.numArt, indice)
                            else:
                                self.ins_dic(field, term, self.numArt)
                            
            else: 
                text = j[self.def_field]
                terms = self.tokenize(text)
                # Insertar los terminos en el diccionario
                for indice, term in enumerate(terms):
                    if self.positional:
                        self.ins_dic_positional(self.def_field, term, self.numArt, indice)
                    else:
                        self.ins_dic(self.def_field, term, self.numArt)
        if self.stemming:  
            self.make_stemming()
        if self.permuterm:
            self.make_permuterm()

    def ins_dic_positional(self, field, key, doc, position):
        """
        Inserta el termino en el diccionario, de manera posicional, es decir, teniendo en cuenta la posición del término en el documento.
        
        input: "field": campo en el cual insertar el documento.
                "key" : término del cual se va a modificar su posting list.
                "doc" : identificador del documento donde aparece el término.
                "posicion" : posición donde aparece el término dentro del documento
        """
        if field not in self.index: # Si el campo no existe en el índice se indexa un diccionario vacío
            self.index[field] = {}
        if key not in self.index[field]: # Si el término no habia sido indexado, se indexa para este una lista que contiene el documento
            self.index[field][key] = {}
        if doc not in self.index[field][key]: # Si el documento no tenía asociada una lista para ese término, se le indexa una vacia.
            self.index[field][key][doc] = []
        self.index[field][key][doc].append(position) # Por último, se inserta para ese field, término y documento, la posición.
            

    def ins_dic(self, field, key:str, doc):
        """
        Inserta el termino en el diccionario, de manera normal. Sin tener en cuenta la posición del término en el documento.
        
        input: "field": campo en el cual insertar el documento.
                "key" : término del cual se va a modificar su posting list.
                "doc" : identificador del documento donde aparece el término.
        """
        if field not in self.index: # Si el campo no existe en el índice se indexa un diccionario vacío
            self.index[field] = {}
        if key not in self.index[field]: # Si el término no habia sido indexado, se indexa para este una lista que contiene el documento
            self.index[field][key] = [doc]
        elif self.index[field][key][-1] != doc: # Si existía y el último documento añadido es distinto del actual, se inserta
            self.index[field][key].append(doc)


    def set_stemming(self, v:bool):

        """
        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.
        """

        self.use_stemming = v


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()


    def make_stemming(self):
        """

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE STEMMING.

        "self.stemmer.stem(token) devuelve el stem del token"


        """
        
        if self.multifield: #si esta el multifield activo 
            multifield = [x for x,y in self.fields if y]
        else:
            multifield = [self.def_field]

        for field in multifield:
            self.sindex[field] = {}
            for word in self.index[field].keys(): #para cada palabra 
                stem = self.stemmer.stem(word) #le aplicamos stemming
                if stem not in self.sindex[field]: #si el stem no estaba indexado previamente, se indexa y se le asocia la lista con la palabra
                    self.sindex[field][stem] = [word]
                else:
                    if word not in self.sindex[field][stem]: #si ya estaba creada la lista del stem, simplemente agregamos la palabra a la lista asociada al stem.
                        self.sindex[field][stem] += [word]
                


    
    def make_permuterm(self):
        """

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE PERMUTERM


        """
        if self.multifield: #si esta el multifield activo 
            multifield = [x for x,y in self.fields if y] # seleccionamos los compos para los que crear indice permuterm
        else:
            multifield = [self.def_field] # por defecto se usa el definido por si no es multifield
        for field in multifield: # para cada campo para el cual es necesario crear el índice permuterm
            self.ptindex[field] = {}
            for token in self.index[field]: # Para cada palabra
                token_p = token + '$' # añadimos al final el símbolo $ y creamos los permuterm moviendo cada letra de una en una de la primera posición a la final
                for _ in range(len(token_p)):
                    token_p = token_p[1:] + token_p[0]
                    if token_p not in self.ptindex[field]:
                        self.ptindex[field][token_p] = token



    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        options = [["TOKENS", "tokens"]]
        if self.multifield:
            stats = [pos[0] for pos in self.fields]
        else: 
            stats =['all']

        print("========================================")
        print("Number of indexed files: "+str(self.numDocs))
        print("----------------------------------------")
        print("Number of indexed articles: "+str(self.numArt))
        print("----------------------------------------")
        for option in options:
            print(option[0]+":")
            for stat in stats:  
                print("\t# of "+option[1]+" in "+"'" +stat+"'"+": " + str(len(self.index[stat])))
            print("---------------------------------------")
        
        if self.permuterm:
            print("PERMUTERMS:")
            for stat in stats:  
                print("\t# of permuterms in "+"'" +stat+"'"+": " + str(len(self.ptindex[stat] if stat in self.ptindex else self.index[stat])))
                # Lo único a destacar es que hemos considerado que como la url no hay que tokenizarla, ni hacerle stemming, ni permuterm
                # es mejor mostrar el valor del campo en el índice normal
            print("---------------------------------------")
        if self.stemming:
            print("STEMS:")
            for stat in stats:  
                print("\t# of stems in "+"'" +stat+"'"+": " + str(len(self.sindex[stat] if stat in self.sindex else self.index[stat])))
                # Lo único a destacar es que hemos considerado que como la url no hay que tokenizarla, ni hacerle stemming, ni permuterm
                # es mejor mostrar el valor del campo en el índice normal
            print("---------------------------------------")


        if self.positional:
            print("Positional queries are allowed")
        else:
            print("Positional queries are NOT allowed")
        print("========================================")

 
    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def query_simple(self, query, ant):
        """
        Resuelve una query.
        
        param:  "query": cadena con la query
                "ant"  : diccionario con los valores de las anteriores consultas realizadas al estar dentro de paréntesis.

        return: posting list con el resultado de la query

        """
        # Primero se procesa la entrada, separando la query en función de las distintas operaciones binarias que se pueden realizar entre posting lists.
        split = list(filter(None,re.split(r'(NOT|AND NOT|AND|OR NOT|OR)', query)))
        # Después se convierten todas las palabras a minúsculas y se eliminan espacios innecesarios.
        split = list(map(str.lower,(map(str.strip, split))))
        indice = 0 # Usado para recorrer la consulta
        # Si la consulta not empieza por not
        if split[indice] != "not":
            # Entonces, si el término estaba en ant, es decir, era una consulta de paréntesis
            if split[indice] in ant:
                # entonces, se recupera su valor
                p1 = ant[split[indice]]
            # Si no, se recupera el field y los terminos para acceder a la posting list.
            else:
                field, term = self.get_field_and_term(split[indice])
                # Se obtiene la posting list del termino en el campo field.
                p1 = self.get_posting(term, field)
        # Si es necesario negarla, se obtiene al igual que en el caso anterior la posting list, salvo que en este caso se negaría usando
        # reverse_posting.
        else: 
            if split[indice + 1] in ant:
                p1 = self.reverse_posting(ant[split[indice + 1]])
            else:
                field, term = self.get_field_and_term(split[indice + 1])     
                p1 = self.get_posting(term, field)
                p1 = self.reverse_posting(p1)
            indice += 1
        indice += 1
        # Una vez se tiene la posting list p1, se va actualizando su valor respecto de la operación y término(o terminos) que se van encontrando
        while indice < len(split) - 1:

            field, term = self.get_field_and_term(split[indice + 1])
            p2 = self.get_posting(term, field) if term not in ant else ant[split[indice + 1]]
            op = split[indice]
            # Se obtiene la nueva posting list en función de la operación a realizar
            if(op == "and"):
                p1 = self.and_posting(p1, p2)
            elif(op == "or"):
                p1 = self.or_posting(p1, p2)
            elif(op == "and not"):
                p1 = self.minus_posting(p1, p2)
            else:
                p1 = self.or_posting(p1, self.reverse_posting(p2))
            indice += 2
        # Al final del todo lo que queda en la variable p1 es el resultado de la query inicial.
        return p1

    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        if query is None or len(query) == 0:
            return []

        # Los siguientes atributos son necesarios para la funcionalidad de paréntesis
        pila = []
        actual = ""
        ant = {}
        numPar = 0
        query = "(" + query + ")"
        # Para cada carácter de la query
        for c in query:
            # Cuando se encuentra un parentesis de apertura se almacena lo que se tenía antes en una pila
            if c == '(':
                pila.append(actual)
                # Y se empieza a leer y almacenar lo que hay dentro del paréntesis en 
                actual = ""

            # Cuando se encuentra un paréntesis de cierre entonces todo lo que había almacenado en la variable actual
            # representa la consulta y se procesa
            elif c == ')':
                # Almacenamos el valor de la consulta a realizar, para luego poder acceder a él
                ant["epccant" + str(numPar)] = self.query_simple(actual, ant)
                # La nueva consulta a seguir realizando será la que estuviese arriba del todo en la pila
                actual = pila.pop()
                # Sustituimos el paréntesis con un término clave para poder acceder posteriormente a su valor
                actual += "epccant" + str(numPar)
                numPar += 1
            else:
                # Mientras sean carácteres normales, se añaden a la variable actual
                actual += c
        return list(ant.values())[-1]

    def get_posting(self, term:str, field:Optional[str]='all'):
        """

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list
        
        NECESARIO PARA TODAS LAS VERSIONES

        """
        

        # Si el término contiene más de una palabra o empieza por " debe considerarse que la consulta es de tipo posicional para ese término
        if len(term.split())> 1 or (self.positional and term[0] == '"'):
            return self.get_positionals(term, field)
        
        #   Añadido de ALT para corregir ortográficamente si se activa la opción 'use_spelling' y la palabra no se encuentra en el índice.
        #   Siempre y cuando la palabra sea un término aislado.
        elif self.use_spelling and term not in self.index[field]:
            res, options = self.speller.suggest(term), []
            # Tomamos la posting list de todas las palabras que sugiere el 'SpellSuggester' y las unimos
            for option in options:
                res = self.or_posting(self.index[field][option],res)
            # Devolvemos el resultado
            return res
        
        # En caso contrario y si hay * o ? se considera que la consulta es de tipo permuterm
        elif re.search("[\*\?]",term):
            return self.get_permuterm(term, field)
        # En caso de que no sea busqueda posicional ni permuterm, si la estructura del índice es posicional
        elif self.positional:
            # Y se debe usar steamming se trata como stemming
            if self.use_stemming:
                return self.get_stemming(term, field)
            # Si no pues se devuelven las claves del diccionario asociado al término, es decir, los documentos donde aparece el término.
            else:
                return list(self.index[field][term].keys()) if term in self.index[field] else []
        # Si la estructura del índice no es posicional, entonces si hay que usar stemming se procesa como stemming
        elif self.use_stemming:
            return self.get_stemming(term, field)
        # Si no se devuelve la entrada del diccionario para la clave = termino
        return self.index[field][term] if term in self.index[field] else []
        
        

    def interseccion_posicional(self, p1, p2, k):
        """
            Devuelve la intersección posicional entre dos posting list, según la distancia de términos k.

            param: "p1" : primera posting list
                   "p2" : segunda posting list
                   "k"  : distancia entre términos para considerar la intersección posicional

            return: posting list resultado de la intersección posicional
        """        
        i1 = i2 = 0
        docsp1 = list(p1.keys())
        docsp2 = list(p2.keys())
        res = {}
        while i1 < len(docsp1) and i2 < len(docsp2):
            if docsp1[i1] == docsp2[i2]:
                listaux = []
                pp1 = pp2 = 0
                valorespp1 = p1[docsp1[i1]]
                valorespp2 = p2[docsp2[i2]]
                while pp1 < len(valorespp1):
                    while pp2 < len(valorespp2):
                        if valorespp2[pp2] > valorespp1[pp1] and valorespp2[pp2] - valorespp1[pp1] <= k:
                            listaux.append(valorespp2[pp2])
                        elif valorespp2[pp2] > valorespp1[pp1]:
                            break
                        pp2 += 1
                    while listaux and abs(listaux[0] - valorespp1[pp1]) > k:
                        listaux.pop(0)
                    for ps in listaux:
                        if docsp1[i1] not in res:
                            res[docsp1[i1]] = []
                        res[docsp1[i1]].append(ps)
                    pp1 += 1
                i1 += 1
                i2 += 1
            elif docsp1[i1] < docsp2[i2]:
                i1 += 1
            else:
                i2 += 1
        return res
        
    def get_positionals(self, terms:str, field):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        res = {}
        terms = terms.replace('"','') # Por si la busqueda posicional tiene dobles comillas (siempre debería ser así)
        terms = terms.split() # Separamos los términos y el problema se reduce a obtener la interseccion_posicional entre los términos de izquierda a derecha
        if terms[0] in self.index[field]:
            res = self.index[field][terms[0]]
            for term in terms[1:]:
                if term in self.index[field]:
                    res =  self.interseccion_posicional(res, self.index[field][term],1)
                else:
                    # Si alguno no existe se sabe que la solución será nula 
                    res = {}
                    break
        return list(res.keys()) # Luego tan solo se devuelve la lista de documentos (las claves del diccionario res) 

    def get_stemming(self, term:str, field: Optional[str]=None):
        """

        Devuelve la posting list asociada al stem de un termino.
        NECESARIO PARA LA AMPLIACION DE STEMMING

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """        
        try:
            # Usando el steammer conseguimos el stem del término
            stem = self.stemmer.stem(term)
            res = []
            for pl in self.sindex[field][stem]:
                # Dependiendo de como esté creado el índice se recuperan los documentos de las palabras asociadas a esde stem
                # de una manera o de otra:
                # Si es posicional se toman las claves del diccionario
                if self.positional:
                    res = self.or_posting(res, list(self.index[field][pl].keys()))
                # Si no directamente se toma el valor para el stem
                else:
                    res = self.or_posting(res, self.index[field][pl])
            return res
        except:
            # Si el stem no estuviese indexado, devolvería la lista vacia
            return []

    def get_permuterm(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        if '*' in term:
            # En caso de que haya que buscar los términos que puedan tener muchos o ningún carácter en la posición del asterisco
            # se genera la consulta(regex) para buscar los permuterm que coinciden
            split = re.split("(\*)",term) # [ca, *, a] -> a$ca.*
            consulta = split[-1] + "\$" + split[0] +".*"
        elif '?' in term:
            # En caso contrario de que solo se deba sustituir los ? por un carácter se genera la regex de la otra manera.
            split = re.split("(\?)", term) # [ca, ?, a] -> a$ca.$ -> Para marcar el final '$'
            consulta = split[-1] + "\$" + split[0] +".$" 
        
        # Se obtienen los permuterms que hacen match con la consulta
        listPer = [permuterm for permuterm in self.ptindex[field].keys() if re.match(consulta, permuterm)] # sacamos los terminos que coincidan con las busqueda
        
        res = [] # posting list resultado
        if listPer:
            for permuterm in listPer:
                word = self.ptindex[field][permuterm]
                # se obtienen los posting list de las palabras que coinciden con el permuterm y se unen a la posting list de manera eficiente
                res = self.or_posting(list(self.index[field].get(word, {}).keys()), res)
        return res



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        # Por teoría de conjuntos neg(B) = E - B
        return self.minus_posting(list(self.articles.keys()), p)




    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        i1 = i2 = 0
        res = []
        while i1 != len(p1) and i2 != len(p2):
            if p1[i1] == p2[i2]:
                res.append(p1[i1])
                i1 += 1
                i2 += 1
            elif p1[i1] < p2[i2]:
                i1 += 1
            else:
                i2 += 1        
        return res



    def or_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 o p2

        """
        i1 = i2 = 0
        res = []
        while i1 != len(p1) and i2 != len(p2):
            if p1[i1] == p2[i2]:
                res.append(p1[i1])
                i1 += 1
                i2 += 1
            elif p1[i1] < p2[i2]:
                res.append(p1[i1])
                i1 += 1
            else:
                res.append(p2[i2])
                i2 += 1
        while i1 != len(p1):
            res.append(p1[i1])
            i1 += 1
        while i2 != len(p2):
            res.append(p2[i2])
            i2 += 1    
        return res
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        i1 = 0
        i2 = 0
        res = []
        while i1 < len(p1) and i2 < len(p2):
            if p1[i1] == p2[i2]:
                i1 += 1
                i2 += 1
            elif p1[i1] < p2[i2]:
                res.append(p1[i1])
                i1 += 1
            else:
                i2 += 1
        while i1 < len(p1):
            res.append(p1[i1])
            i1 += 1
        return res
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################

    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True                    
            else:
                print(line)
        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        results = self.solve_query(query)        

        print('Query: \'{}\''.format(query))
        print('========================================')
        i = 1
        for result in results:
            documento, posicion = self.articles[result]
            
            with open(self.docs[documento], 'r') as f:
                index = 0
                for line in f:
                    if index == posicion:
                        aux = json.loads(line)
                        print(f"#{i}\t ({result}) \t {aux['title']}: \t url: {aux['url']}")
                        # Si se deben mostrar snippets y la consulta es sencilla
                        if self.show_snippet:
                            print("\t" + self.get_snippet(query, aux))
                        break
                    index += 1
            if not(self.show_all) and i == self.SHOW_MAX:
                break
            i += 1
        
        print('========================================')
        print('Number of results: {}'.format(len(results)))


    def get_field_and_term(self, query):
        """
        Devuelve el field donde realizar la consulta y los terminos a consultar 

        param:  "query": query que debe separar en field y termino/s.

        return: el campo en el que realizar la consulta y la consulta a realizar ahí
        """
        split = re.split(":", query) # separamos el field de los términos
        # si existia field, se usa, sino se usa all como field
        return split[0] if len(split) > 1 else 'all', split[1] if len(split) > 1 else split[0]
    
    def get_snippet(self, query, articulo):
        """
        Devuelve el snippet de una consulta.

        param: "query" : consulta realizada
               "articulo": el diccionario que representa la información del artiículo
    
        return: snippet de la consulta sobre el artículo
        """
        query = re.sub("\(|\)", "", query)
        terminos = list(map(str.lower,map(str.strip,filter(None,re.split(r'(NOT|AND NOT|AND|OR NOT|OR)', query)))))
        indice= 0
        snippet = ""
        while indice < len(terminos):
            # Si la operacion es de negación, no se procesa esa palabra
            if terminos[indice] in ['not', 'and not', 'or not']:
                snippet += "No hay snnipet para el término " + "\033[91m"+ terminos[indice + 1] + "\033[0m por estar negado.\n\t"
                indice += 2
                continue
            # Si no es de negación se sigue adelante
            elif terminos[indice] in ['or', 'and']:
                indice += 1
                continue
            # Si se encuentra un término se procesa y se devuelve el contexto del campo en el que se consulta
            else:
                field, termino = self.get_field_and_term(terminos[indice])
                encontrado = False
                # Si el campo es summary, se debe tomar la ventana, ya que el resumen puede ser demasiado largo
                if field == 'summary':
                    snippet += self.get_ventana(articulo[field], termino, 5) + "\n\t"
                    encontrado = True
                # Si el campo es url o title, se accede directamente al campo
                elif field in articulo:            
                    # Para el termino de la consulta se colorea en rojo en el campo
                    snippet += re.sub("\W+" + termino + "\W+", f' \033[91m{termino}\033[0m ',articulo[field],flags=re.IGNORECASE) + "\n\t"
                    encontrado = True
                elif field == "section-name":
                    # Si el campo era section-name, este no se encuentra en articulo y se debe procesar de la siguiente manera
                    for section in articulo['sections']:
                        # Se busca la sección cuyo nombre contenga los términos
                        if re.search("\W+" + termino + "\W+", section['name'], re.IGNORECASE): 
                            # y se procede de la misma manera, se colorea y se devuelve el snippet
                            snippet += re.sub("\W+" + termino + "\W+", f' \033[91m{termino}\033[0m ',section['name'],flags=re.IGNORECASE) + "\n\t"
                            encontrado = True
                            break
                elif field == "all":
                    # Por último, si se debe buscar entre todo el texto se recorren los diferentes campos, las secciones y subsecciones hasta encontrar la primera ocurrencia.
                    # Una vez encontrada se obtiene la ventana con k = 5 alrededor de ese término y se devuelve.
                    for field in list(articulo.keys())[:-1]:
                        if re.search(" " + termino + " ", articulo[field], re.IGNORECASE):
                            # Si el campo es summary, se debe tomar la ventana, ya que el resumen puede ser demasiado largo
                            if field == 'summary':
                                snippet +=  self.get_ventana(articulo[field], termino, 5) + "\n\t"
                                encontrado = True
                                break
                    if not encontrado:
                        for section in articulo['sections']:
                            if encontrado:
                                break
                            if re.search("\W+" + termino + "\W+", section['name'], flags=re.IGNORECASE):
                                snippet += re.sub(" " + termino + " ", f' \033[91m{termino}\033[0m ',section['name'],flags=re.IGNORECASE) + "\n\t"
                                encontrado = True
                                break
                            if re.search("\W+" + termino + "\W+", section['text'], flags=re.IGNORECASE):
                                snippet += self.get_ventana(section['text'], termino, 5) + "\n\t"
                                encontrado = True
                                break
                            for subsection in section['subsections']:
                                if re.search("\W+" + termino + "\W+", subsection['name'], flags=re.IGNORECASE):
                                    snippet += re.sub(" " + termino + " ", f' \033[91m{termino}\033[0m ',subsection['name'],flags=re.IGNORECASE) + "\n\t"
                                    encontrado = True
                                    break
                                if re.search("\W+" + termino + "\W+", subsection['text'], flags=re.IGNORECASE):
                                    snippet += self.get_ventana(subsection['text'], termino, 5) + "\n\t"
                                    encontrado = True
                                    break
                if not encontrado:
                    snippet += "No aparece el término " + "\033[91m"+ termino +"\033[0m" + "\n\t"
                indice += 1
        return snippet


    def get_ventana(self, text, termino, k):
        """
        Devuelve una ventana de 2*k + 1 palabras, de manera que la palabra central es la primera palabra que coincide con la consulta.

        param: "text"  : Texto donde buscar los término de la consulta.
               "termino" : Termino del cual generar el contexto.
               "k"     : Parámetro que marca el tamaño de la ventana.
        """
        # Preprocesamos el texto
        text = text.replace("\n", " ")
        # Generamos la regex que nos devuelve el contexto de la palabra
        pattern = r"\b(?:\w+\.*\w*\W+){0," + str(k) + r"}" + termino + r"(?:\W+\w+\.*\w*){0," + str(k) + r"}\b"
        # Buscamos el contexto de la primera ocurrencia del término
        matches = re.search(pattern, text, flags=re.IGNORECASE)
        # Obtenemos el snippet y coloreamos el término de rojo
        snippet = matches.group(0)
        snippet = re.sub(termino, f"\033[91m{termino}\033[0m", snippet, flags=re.IGNORECASE)
        return snippet