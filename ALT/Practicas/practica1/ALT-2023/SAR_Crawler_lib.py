#! -*- encoding: utf8 -*-
import heapq as hq

from typing import Tuple, List, Optional, Dict, Union

import requests
import bs4
import re
from urllib.parse import urljoin
import json
import math
import os

class SAR_Wiki_Crawler:

    def __init__(self):
        # Expresión regular para detectar si es un enlace de la Wikipedia
        self.wiki_re = re.compile(r"(http(s)?:\/\/(es)\.wikipedia\.org)?\/wiki\/[\w\/_\(\)\%]+")
        # Expresión regular para limpiar anclas de editar
        self.edit_re = re.compile(r"\[(editar)\]")
        # Formato para cada nivel de sección
        self.section_format = {
            "h1": "##{}##",
            "h2": "=={}==",
            "h3": "--{}--"
        }

        # Expresiones regulares útiles para el parseo del documento
        self.title_sum_re = re.compile(r"##(?P<title>.+)##\n(?P<summary>((?!==.+==).+|\n)+)(?P<rest>(.+|\n)*)")
        self.sections_re = re.compile(r"==[^=\n]+==\n")
        self.subsections_re = re.compile(r"--[^-\n]+--\n")
        
        #self.section_re = re.compile(r"==(?P<name>.+)==\n(?P<text>((?!--.+--).+|\n)*)(?P<rest>(.+|\n)*)")
        #self.section_section_re = re.compile(r"==(?P<name>.+)==\n(?P<text>((?!==[^=\n]+==).+|\n)*)(?P<rest>(.+|\n)*)")
        #self.subsection_re = re.compile(r"--(?P<name>.+)--\n(?P<text>(.+|\n)*)")
        #self.subsection_subsection_re = re.compile(r"--(?P<name>.+)--\n(?P<text>((?!--.+--).+|\n)*)(?P<rest>(.+|\n)*)")


    def is_valid_url(self, url: str) -> bool:
        """Verifica si es una dirección válida para indexar

        Args:
            url (str): Dirección a verificar

        Returns:
            bool: True si es valida, en caso contrario False
        """
        return self.wiki_re.fullmatch(url) is not None


    def get_wikipedia_entry_content(self, url: str) -> Optional[Tuple[str, List[str]]]:
        """Devuelve el texto en crudo y los enlaces de un artículo de la wikipedia
        Args:
            url (str): Enlace a un artículo de la Wikipedia

        Returns:
            Optional[Tuple[str, List[str]]]: Si es un enlace correcto a un artículo
                de la Wikipedia en inglés o castellano, devolverá el texto y los
                enlaces que contiene la página.

        Raises:
            ValueError: En caso de que no sea un enlace a un artículo de la Wikipedia
                en inglés o español
        """
        if not self.is_valid_url(url):
            raise ValueError((
                f"El enlace '{url}' no es un artículo de la Wikipedia en español"
            ))

        try:
            req = requests.get(url)
        except Exception as ex:
            print(f"ERROR: - {url} - {ex}")
            return None


        # Solo devolvemos el resultado si la petición ha sido correcta
        if req.status_code == 200:
            soup = bs4.BeautifulSoup(req.text, "lxml")
            urls = set()

            for ele in soup.select((
                'div#catlinks, div.printfooter, div.mw-authority-control'
            )):
                ele.decompose()

            # Recogemos todos los enlaces del contenido del artículo
            for a in soup.select("div#bodyContent a", href=True):
                href = a.get("href")
                if href is not None:
                    urls.add(href)

            # Contenido del artículo
            content = soup.select((
                "h1.firstHeading,"
                "div#mw-content-text h2,"
                "div#mw-content-text h3,"
                "div#mw-content-text h4,"
                "div#mw-content-text p,"
                "div#mw-content-text ul,"
                "div#mw-content-text li,"
                "div#mw-content-text span"
            ))

            dedup_content = []
            seen = set()

            for element in content:
                if element in seen:
                    continue

                dedup_content.append(element)

                # Añadimos a vistos, tanto el elemento como sus descendientes
                for desc in element.descendants:
                    seen.add(desc)

                seen.add(element)

            text = "\n".join(
                self.section_format.get(element.name, "{}").format(element.text)
                for element in dedup_content
            )

            # Eliminamos el texto de las anclas de editar
            text = self.edit_re.sub('', text)

            return text, sorted(list(urls))

        return None


    def parse_wikipedia_textual_content(self, text: str, url: str) -> Optional[Dict[str, Union[str,List]]]:
        """Devuelve una estructura tipo artículo a partir del text en crudo

        Args:
            text (str): Texto en crudo del artículo de la Wikipedia
            url (str): url del artículo, para añadirlo como un campo

        Returns:

            Optional[Dict[str, Union[str,List[Dict[str,Union[str,List[str,str]]]]]]]:

            devuelve un diccionario con las claves 'url', 'title', 'summary', 'sections':
                Los valores asociados a 'url', 'title' y 'summary' son cadenas,
                el valor asociado a 'sections' es una lista de posibles secciones.
                    Cada sección es un diccionario con 'name', 'text' y 'subsections',
                        los valores asociados a 'name' y 'text' son cadenas y,
                        el valor asociado a 'subsections' es una lista de posibles subsecciones
                        en forma de diccionario con 'name' y 'text'.

            en caso de no encontrar título o resúmen del artículo, devolverá None

        """
        def clean_text(txt):
            return '\n'.join(l for l in txt.split('\n') if len(l) > 0)

        #Usar una expresion regular para comprobar si hay titulo, summary y resto del texto
        info = self.title_sum_re.findall(text)

        # Si no hay titulo o summary devolver None 
        if(len(info) == 0):
            return None
        if(len(info[0][1]) <= 4):
            return None

        # guardar la informacion basica del documento de la wikipedia
        document = {
            'url': url,
            'title': info[0][0],
            'summary': info[0][1],
            'sections': []}  
        rest = info[0][3]
        
        # obtenemos los nombres de las secciones y sus textos 
        sections = self.sections_re.findall(rest)
        sectionsText = self.sections_re.split(rest)[1:] 
        for s in range(len(sections)):
            #guardamos todo el texto de la seccion actual
            allThisSectionText = sectionsText[s] 
            
            # obtenemos las subsecciones de la seccion, para en caso de no tener guardar el texto directamente 
            subsections = self.subsections_re.findall(allThisSectionText)
            if(len(subsections)==0):
                sect = self.crear_sec(sections[s][2:-3], allThisSectionText)
                document["sections"].append(sect)
               # print(sect) 
                continue
            else:
                 # y en caso de tener subseccioens guardamos el texto tras separarlo
                subsectionsText = self.subsections_re.split(allThisSectionText)
                sect = self.crear_sec(sections[s][2:-3],subsectionsText[0])
                document["sections"].append(sect)
              
            # guardamos las subsecciones con su correspondientes textos
            for subs in range(1,len(subsectionsText)):
                subSec  = {
                    "name": subsections[subs-1][2:-3],
                    "text": subsectionsText[subs]    
                }
                document["sections"][s]["subsections"].append(subSec)      
        return document

    def crear_sec(self, name, text):
        sec = {
        "name": name,
        "text": text,
        "subsections": []} 
        return sec


    def save_documents(self,
        documents: List[dict], base_filename: str,
        num_file: Optional[int] = None, total_files: Optional[int] = None
    ):
        """Guarda una lista de documentos (text, url) en un fichero tipo json lines
        (.json). El nombre del fichero se autogenera en base al base_filename,
        el num_file y total_files. Si num_file o total_files es None, entonces el
        fichero de salida es el base_filename.

        Args:
            documents (List[dict]): Lista de documentos.
            base_filename (str): Nombre base del fichero de guardado.
            num_file (Optional[int], optional):
                Posición numérica del fichero a escribir. (None por defecto)
            total_files (Optional[int], optional):
                Cantidad de ficheros que se espera escribir. (None por defecto)
        """
        assert base_filename.endswith(".json")

        if num_file is not None and total_files is not None:
            # Separamos el nombre del fichero y la extensión
            base, ext = os.path.splitext(base_filename)
            # Padding que vamos a tener en los números
            padding = len(str(total_files))

            out_filename = f"{base}_{num_file:0{padding}d}_{total_files}{ext}"

        else:
            out_filename = base_filename

        with open(out_filename, "w", encoding="utf-8", newline="\n") as ofile:
            for doc in documents:
                print(json.dumps(doc, ensure_ascii=True), file=ofile)


    def start_crawling(self, 
                    initial_urls: List[str], document_limit: int,
                    base_filename: str, batch_size: Optional[int], max_depth_level: int,
                    ):        

        """Comienza la captura de entradas de la Wikipedia a partir de una lista de urls válidas, 
            termina cuando no hay urls en la cola o llega al máximo de documentos a capturar.
        
        Args:
            initial_urls: Direcciones a artículos de la Wikipedia
            document_limit (int): Máximo número de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.
            max_depth_level (int): Profundidad máxima de captura.
        """

        # URLs válidas, ya visitadas (se hayan procesado, o no, correctamente)
        visitedList = [] 
        visited = set()
        # URLs en cola
        to_process = set(initial_urls)
        # Direcciones a visitar
        queue = [(0,"",url) for url in to_process]
        hq.heapify(queue)
        # Buffer de documentos capturados
        documents: List[dict] = []
        # Contador del número de documentos capturados
        total_documents_captured = 0
        # Contador del número de ficheros escritos
        files_count = 1
        # Contador del número de documentos capturados de la batch actual 
        currentBatch = 0

        # En caso de que no utilicemos bach_size, asignamos None a total_files
        # así el guardado no modificará el nombre del fichero base
        if batch_size is None:
            total_files = None
        else:
            # Suponemos que vamos a poder alcanzar el límite para la nomenclatura
            # de guardado
            total_files = math.ceil(document_limit / batch_size)

        for pack in queue:
            # Actualizar el contador de documentos totales 
            total_documents_captured += 1

            # Separar la profundiad y url
            depth = pack[0] 
            url = pack[2] 

            # Actualizar lista de paginas visitadas  
            visitedList.append(url)
            visited = set(visitedList)

            # En caso de haber procesado todos los documentos, o de que la profundiad se mayor a la permitida
            # guardar los documentos en el siguiente fichero y parar
            if (total_documents_captured > document_limit):
                self.save_documents(documents, base_filename, files_count, total_files)
                break

            #FUNCIONAMIENTO BASICO# 
            # Actualizar el contador de documentos de la batch actual 
            currentBatch += 1

            # Obtener el texto y la lista de urls de la pagina 
            text, urlList = self.get_wikipedia_entry_content(url)

            # guardar los links solo si estamos en una profundidad inferior a la maxima
            if depth < max_depth_level:
            # Anadir los url validos a la lista
                for u in urlList:
                    # Para anadir tiene que no se haber sido visitado y ser una url valida 
                    if (u not in visited and self.is_valid_url(u)):                
                        queue.append((depth+1,"","https://es.wikipedia.org"+u))

            # procesar el texto y anadirlo el diccionario resultante a la siguiente batch
            documents.append(self.parse_wikipedia_textual_content(text , url))

            # En caso de alcanzar el tamano de una batch guardarla en el fichero, actualizar contadores y seguir
            if currentBatch == batch_size:
                currentBatch = 0
                self.save_documents(documents, base_filename, files_count, total_files)
                documents: List[dict] = []
                files_count += 1
                
            


    def wikipedia_crawling_from_url(self,
        initial_url: str, document_limit: int, base_filename: str,
        batch_size: Optional[int], max_depth_level: int
    ):
        """Captura un conjunto de entradas de la Wikipedia, hasta terminar
        o llegar al máximo de documentos a capturar.
        
        Args:
            initial_url (str): Dirección a un artículo de la Wikipedia
            document_limit (int): Máximo número de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.
            max_depth_level (int): Profundidad máxima de captura.
        """
        if not self.is_valid_url(initial_url) and not initial_url.startswith("/wiki/"):
            raise ValueError(
                "Es necesario partir de un artículo de la Wikipedia en español"
            )

        self.start_crawling(initial_urls=[initial_url], document_limit=document_limit, base_filename=base_filename,
                            batch_size=batch_size, max_depth_level=max_depth_level)



    def wikipedia_crawling_from_url_list(self,
        urls_filename: str, document_limit: int, base_filename: str,
        batch_size: Optional[int]):
        """A partir de un fichero de direcciones, captura todas aquellas que sean
        artículos de la Wikipedia válidos

        Args:
            urls_filename (str): Lista de direcciones
            document_limit (int): Límite máximo de documentos a capturar
            base_filename (str): Nombre base del fichero de guardado.
            batch_size (Optional[int]): Cada cuantos documentos se guardan en
                fichero. Si se asigna None, se guardará al finalizar la captura.

        """

        urls = []
        with open(urls_filename, "r", encoding="utf-8") as ifile:
            for url in ifile:
                url = url.strip()

                # Comprobamos si es una dirección a un artículo de la Wikipedia
                if self.is_valid_url(url):
                    if not url.startswith("http"):
                        raise ValueError(
                            "El fichero debe contener URLs absolutas"
                        )

                    urls.append(url)

        urls = list(set(urls)) # eliminamos posibles duplicados

        self.start_crawling(initial_urls=urls, document_limit=document_limit, base_filename=base_filename,
                            batch_size=batch_size, max_depth_level=0)





if __name__ == "__main__":
    raise Exception(
        "Esto es una librería y no se puede usar como fichero ejecutable"
    )