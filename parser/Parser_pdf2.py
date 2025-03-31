import re
import unicodedata
from PyPDF2 import PdfReader
import spacy
from parser.Chunking_loading import create_table, insert_chunks, create_conn

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

def extraer_texto(pdf_path:str, i ) -> str:
    """
    Extrae el texto completo de un PDF.

    Args:
        pdf_path (str): Ruta al archivo PDF.
    
    Returns:
        str: Texto extraido del PDF
    """
    lector = PdfReader(pdf_path)
    texto = ""
    titulo = ""
    for pagina in lector.pages:
        # si la pagina contiene indice, buscamos el texto escrito antes de indice para conseguir el titulo
        for t in ["índice", "tabla de contenidos"]:
            if t in pagina.extract_text().lower():
                titulo = pagina.extract_text().strip().lower().split(t)[0]
                titulo = re.sub(r'\s+', ' ', titulo).strip()
        texto += pagina.extract_text() + "\n"
    titulo = normalize_text(titulo)
    return texto, titulo

import re

def eliminar_indice(texto: str, titulo: str) -> str:
    """
    Elimina la sección de índice del texto, sin asumir que el encabezado del índice
    aparezca al comienzo de la línea. Se revisa el primer 20% de líneas y se eliminan aquellas
    que contengan palabras clave propias del índice (como "índice", "tabla de contenidos", "anexos")
    o que parezcan entradas de índice (por ejemplo, terminan en un número de página).
    El índice termina cuando el título proporcionado aparece después de la inicialización.
    
    Args:
        texto (str): Texto completo del documento.
        titulo (str): El título que marca el final del índice.

    Returns:
        str: Texto con la sección de índice eliminada.
    """
    inicio = ["índice","indice","tabla de contenidos"]
    for t in inicio:
        if t in texto.lower():
            inicio = t
            break


    lines = texto.splitlines()
    new_lines = []
    in_index = False

    # Pattern para detectar el inicio del índice (cuando "Índice" o "Tabla de Contenidos" aparece).
    index_start_pattern = re.compile(r'(?i)^(índice|indice|tabla de contenidos)')
    
    # Patrón para detectar entradas del índice: líneas que terminan en números de página.
    index_line_pattern = re.compile(r'.*\s+\d+\s*$|.*\s*\d+\s*$')

    index_upper_char = re.compile(r'^[A-Z][a-z].*$|^[A-Z]\s+.*$|[ÁÉÍÓÚ].*$') 

    index_all_upper = re.compile(r'^[A-Z].*$')
    # Patrón para detectar un encabezado real que inicia el contenido.
    # Puede ser un número, romano o la palabra "anexos" (considerando encabezados numerados).
    header_pattern = re.compile(r'^(?:\d+\.\d*\.*\s*$|[ivxlcdm]+\.\s*|[IVXLCDM]+\.\s*|anexos\s+)', re.IGNORECASE)

    index_stopper_char = re.compile(r'●.*$|-.*$|Página.*$|P á g i n a.*$')
    
    header_pattern2 = re.compile(r'^\d+(\.\d+)*\.\s*$|ANEXOS|anexos\s+|ANEXO')
    first_index = None
    indexes = []
    in_text = False
    index_ = ""
    flg_upper = False
    last_index = '0'
    cnt_mayus = 0
    for line in lines:
        
        # Si encontramos la línea que inicia el índice, activamos el flag
        if not in_index and index_start_pattern.search(line):
            inicio = []
            in_index = True
            continue
        
        if in_index:
            # Si la línea parece una entrada del índice (por ejemplo, termina en un número), la omitimos
            if index_line_pattern.match(line):
                continue
            
            if header_pattern.match(line):
                if first_index is None:
                    first_index = line
                else:
                    if first_index == line:
                        in_index = False
                        in_text = True
                        new_lines.append(line)
                continue
            

            # Si la línea parece un encabezado (número o romano) y NO contiene un número de página, 
            # asumimos que es el inicio del contenido real.
            # verificamos si la linea es una subcadena del titulo para evitar errores

            if not index_line_pattern.match(line) and first_index is None:
                in_index = False
                in_text = True
                new_lines.append(line)  # Se agrega la primera línea del contenido
                continue
            
            # Si estamos dentro del índice, omitimos esta línea
            continue
        else:
            if type(inicio) != type([]):
                continue
            new_lines.append(line)

        if in_text:
                    

            if header_pattern2.match(line):
                try:
                    tmp_idx = float(last_index.split('.')[0])
                    tmp_line = float(line.split('.')[0])
                    if abs(tmp_idx - tmp_line) <= 1:                    
                        last_index = line
                        if index_ != "":
                            cnt_mayus = 0
                        if len(index_) > 6 and len(index_) < 80:
                                indexes.append(index_)
                        index_ = line
                        flg_upper = False
                    continue
                except:
                    if index_ == "":
                        index_ = line
                        continue

            if index_ != "":
                if len(index_) > 15:
                    cnt_mayus += 1

                if (index_upper_char.match(line) or index_all_upper.match(line)) and flg_upper == False:
                    cnt_mayus += 1
                    flg_upper = True
                    index_ += " " + line
                    continue
                elif (index_upper_char.match(line) or index_all_upper.match(line)) and flg_upper == True and cnt_mayus == 1:
                    cnt_mayus += 1
                elif (index_upper_char.match(line) and flg_upper == True and cnt_mayus > 1) or index_stopper_char.match(line):
                    flg_upper = False
                    cnt_mayus = 0
                    if len(index_) > 6 and len(index_) < 80:
                        indexes.append(index_)
                    index_ = ""
                if flg_upper == True:
                    index_ += " " + line

 
    with open("index.txt", "w", encoding="utf-8") as f:
        for i in indexes:
            f.write(i + "\n")
    for i in range(len(indexes)):
        indexes[i] = normalize_text(indexes[i])
        indexes[i] = remove_connector_words(indexes[i])
    import time
    time.sleep(9)
    texto_limpio = "\n".join(new_lines)
    
    texto_limpio = normalize_text(texto_limpio)
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()

    texto_limpio = re.sub(titulo, '', texto_limpio, flags=re.IGNORECASE)
    return texto_limpio, indexes


def remove_connector_words(texto: str, connector_words: dict=None)-> str:
    """
    Elimina palabras conectivas (ej: y, o, pero, etc.) del texto.

    Args:
        texto (str): Texto del documento.
        connector_words (dict): Conjunto de palabras conectivas a eliminar.

    Returns:
        str: Texto sin las palabras conectivas.
    """
    if connector_words is None:
        connector_words = {"y", "o", "ni", "pero", "sino", "aunque", 
                           "ademas", "tampoco", "sin", "embargo", "no obstante", "aun", "de"}
    pattern = r'\b(' + '|'.join(re.escape(word) for word in connector_words) + r')\b'
    texto_sin_conectores = re.sub(pattern, '', texto, flags=re.IGNORECASE)
    texto_sin_conectores = re.sub(r'\s+', ' ', texto_sin_conectores)
    return texto_sin_conectores.strip()

def normalize_text(texto: str) -> str:
    """
    Normaliza el texto:
      - Convierte a minusculas.
      - Elimina tildes y otros diacriticos.
      - Elimina signos de puntuacion.

    Args:
        texto (str): Texto del documento.
    
    Returns:
        str: Texto normalizado.
    """
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto)
    texto = texto.encode('ASCII', 'ignore').decode('utf-8')
    #texto = re.sub(r'[^\w\s]', '', texto)
    return texto

def remove_pagination_words(texto: str) -> str:
    """
    Elimina palabras que tengan el formato "pagina1de2" o que contengan la cadena "'n'de'm'".

    Args:
        texto (str): Texto del documento.
    
    Returns:
        str: Texto sin las palabras de paginacion.
    """
    texto = re.sub(r'\bpagina\d+de\d+\b', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\bp a g i n a\d+d e\d+\b', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\b\w*\d+de\d+\w*\b', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto




