import re
import unicodedata
from PyPDF2 import PdfReader
import spacy

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

def extraer_texto(pdf_path:str) -> str:
    """
    Extrae el texto completo de un PDF.

    Args:
        pdf_path (str): Ruta al archivo PDF.
    
    Returns:
        str: Texto extraido del PDF
    """
    lector = PdfReader(pdf_path)
    texto = ""
    for pagina in lector.pages:
        texto += pagina.extract_text() + "\n"
    return texto

def eliminar_indice(texto: str) -> str:
    """
    Elimina la seccion del indice (tabla de contenidos) del texto.
    Se asume que la seccion de indice comienza con palabras clave como 'Indice' o 'Tabla de Contenidos'.

    Args:
        texto (str): Texto del documento.
    
    Returns:
        str: Texto sin la seccion de indice.
    """
    patron = r'(Índice|Tabla de Contenidos).*?(?=\n\S)'
    texto_limpio = re.sub(patron, '', texto, flags=re.DOTALL | re.IGNORECASE)
    return texto_limpio

def renumerar_secciones(texto: str) -> str:
    """
    Reenumera los encabezados del documento.
    Se asume que los encabezados comienzan con numeros seguidos de un punto (ej: "1. Introduccion", "1.1. Alcance").
    Se detecta el nivel de la seccion por la cantidad de puntos y se reemplaza con una numeración secuencial.

    Args:
        texto (str): Texto del documento.
    
    Returns:
        str: Texto con los encabezados renumerados.
    """
    lineas = texto.splitlines()
    nuevas_lineas = []
    contador_seccion = 1
    contador_subseccion = 1

    for linea in lineas:
        if re.match(r'^\d+(\.\d+)*\s', linea):
            nivel = linea.split()[0].count('.')
            if nivel == 1:
                nuevo_numero = f"{contador_seccion}."
                contador_seccion += 1
                contador_subseccion = 1
            elif nivel == 2:
                nuevo_numero = f"{contador_seccion - 1}.{contador_subseccion}"
                contador_subseccion += 1
            else:
                nuevo_numero = linea.split()[0]
            nueva_linea = re.sub(r'^\d+(\.\d+)*', nuevo_numero, linea)
            nuevas_lineas.append(nueva_linea)
        else:
            nuevas_lineas.append(linea)
    return "\n".join(nuevas_lineas)

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
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
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

def lemmatize_text(texto: str) -> str:
    """
    Aplica lematizacion al texto, transformando verbos a infinitivo y
    algunos pronombres plurales a su forma singular (por ejemplo, "nosotros" -> "yo").

    Args:
        texto (str): Texto del documento.

    Returns:
        str: Texto lematizado.
    """
    pronoun_mapping = {
        "nosotros": "yo",
        "nosotras": "yo",
        "vosotros": "tu",
        "vosotras": "tu",
        "ellos": "el",
        "ellas": "ella"
    }
    doc = nlp(texto)
    lemmatized_tokens = []
    for token in doc:
        lemma = token.lemma_
        if token.pos_ == "PRON" and lemma in pronoun_mapping:
            lemma = pronoun_mapping[lemma]
        lemmatized_tokens.append(lemma)
    return " ".join(lemmatized_tokens)

def chunk_text(texto: str, chunk_size: int=200, overlap: int=25) -> list:
    """
    Divide el texto en fragmentos de aproximadamente 'chunk_size' palabras con un solapamiento de 'overlap' palabras.
    Esto es util para sistemas RAG que requieren fragmentos manejables para indexacion y busqueda.

    Args:
        texto (str): Texto del documento.
        chunk_size (int): Tamanho de los fragmentos.
        overlap (int): Cantidad de palabras de solapamiento entre fragmentos.

    Returns:
        list: Lista de fragmentos (chunks) del texto.
    """
    palabras = texto.split()
    chunks = []
    inicio = 0
    while inicio < len(palabras):
        fin = min(inicio + chunk_size, len(palabras))
        chunk = " ".join(palabras[inicio:fin])
        chunks.append(chunk)
        inicio += chunk_size - overlap
    return chunks

def parser_uniformizador(pdf_path1: str, pdf_path2: str, salida_base: str) -> None:
    """
    Procesa dos PDFs:
      - Extrae el texto.
      - Elimina el indice.
      - Renumera las secciones.
      - Elimina palabras conectivas.
      - Normaliza el texto (minusculas, eliminacion de tildes y signos de puntuacion).
      - Elimina palabras con formato "pagina1de2" o que contengan "ndem".
      - Aplica lematización: transforma verbos a infinitivo y pronombres plurales a singular (según mapeo).
      - Divide el contenido en fragmentos (chunks) para uso en sistemas RAG.
      
    Se generan archivos de salida para cada PDF:
      - Un archivo con el texto uniformizado.
      - Un archivo con los chunks, separados por una línea delimitadora.

    Args:
        pdf_path1 (str): Ruta al primer PDF.
        pdf_path2 (str): Ruta al segundo PDF.
        salida_base (str): Nombre base para los archivos de salida. Se anhade sufijo "_1" y "_2" para cada PDF
    
    Returns:
        None
    """
    texto1 = extraer_texto(pdf_path1)
    texto2 = extraer_texto(pdf_path2)
    
    texto1 = eliminar_indice(texto1)
    texto2 = eliminar_indice(texto2)
    
    texto1 = renumerar_secciones(texto1)
    texto2 = renumerar_secciones(texto2)
    
    texto1 = remove_connector_words(texto1)
    texto2 = remove_connector_words(texto2)
    
    texto1 = normalize_text(texto1)
    texto2 = normalize_text(texto2)
    
    texto1 = remove_pagination_words(texto1)
    texto2 = remove_pagination_words(texto2)
    
    texto1 = lemmatize_text(texto1)
    texto2 = lemmatize_text(texto2)
    
    chunks1 = chunk_text(texto1)
    chunks2 = chunk_text(texto2)
    
    with open(salida_base + "_1.txt", "w", encoding="utf-8") as f:
        f.write(texto1)
    with open(salida_base + "_2.txt", "w", encoding="utf-8") as f:
        f.write(texto2)
    
    with open(salida_base + "_1_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks1:
            f.write(chunk + "\n")
    with open(salida_base + "_2_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks2:
            f.write(chunk + "\n")
    
    print("Proceso completado. Archivos guardados:")
    print("Texto uniformizado:", salida_base + "_1.txt", "y", salida_base + "_2.txt")
    print("Chunks para RAG:", salida_base + "_1_chunks.txt", "y", salida_base + "_2_chunks.txt")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Uso: python parser.py archivo1.pdf archivo2.pdf salida_base")
    else:
        parser_uniformizador(sys.argv[1], sys.argv[2], sys.argv[3])
