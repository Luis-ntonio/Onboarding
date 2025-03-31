import math
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import numpy as np
from db.connection import model

def retrieve_knn(conn, query_text, k=5):
    """
    Dado un query, se obtiene su embedding y se recuperan los K chunks más similares usando la busqueda de vecinos.
    """
    cur = conn.cursor()
    query_embedding = model.encode(query_text).tolist()
    # La consulta utiliza el operador <-> (distancia euclidiana o coseno, según la configuración de PGVector)
    sql = """
        SELECT name, indexes, text, 
               embedding <-> %s::vector AS distance,
               similarity(indexes::text, %s::text) AS text_similarity
        FROM chunks
        WHERE similarity(indexes::text, %s::text) > 0
        ORDER BY text_similarity DESC
        LIMIT %s;
    """

    cur.execute(sql, (query_embedding, query_text, query_text, k))
    results = cur.fetchall()
    cur.close()
    return results

def chunk_text(texto: str, indices: list[str], chunk_size: int=200, overlap: int=25) -> list:
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
    indexes_used = []
    inicio = 0
    last_index = ""

    while inicio < len(palabras):
        index_ = []
        if last_index != "":
            index_.append(last_index)

        fin = min(inicio + chunk_size, len(palabras))
        chunk = " ".join(palabras[inicio:fin])
        for index in indices:
            if index in chunk:
                last_index = index
                index_.append(index)

        indexes_used.append(index_)
        chunks.append(chunk)
        inicio += chunk_size - overlap
    return chunks, indexes_used

def chunk_segment(words: list[str], n_chunks: int, chunk_size: int, overlap: int) -> list[str]:
    """
    Divide una lista de palabras en n_chunks, usando un tamaño de chunk y un solapamiento.
    Si n_chunks es 1, retorna todo el segmento.
    
    Se utiliza la fórmula:
        step = (total_words - overlap) / (n_chunks - 1)
    y para cada chunk se toma desde start = round(i * step) hasta start+chunk_size (o hasta el final).
    """
    if n_chunks <= 1:
        return [" ".join(words)]
    
    total = len(words)
    step = (total - overlap) / (n_chunks - 1)
    chunks = []
    for i in range(n_chunks):
        start = int(round(i * step))
        end = start + chunk_size
        if end > total:
            end = total
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
    return chunks

def chunk_text_indexes(texto1: str, texto2: str, indices: list[str], 
                       chunk_size: int = 200, overlap: int = 25) -> tuple[list[str], list[str], list[str]]:
    """
    Divide el texto en fragmentos (chunks) utilizando los índices como puntos de corte.
    
    Para cada marcador en 'indices', se extrae el segmento de cada texto (texto1 y texto2)
    que va desde ese índice hasta el siguiente (o hasta el final).
    
    Si el segmento tiene menos de 200 caracteres, se conserva como un único chunk.
    Si es superior, se divide en chunks de aproximadamente 'chunk_size' palabras con 'overlap' palabras de solapamiento.
    
    Además, se fuerza que el número de chunks para cada índice sea el mismo en texto1 y texto2,
    re-dividiendo (si es necesario) el segmento de menor número de chunks para igualarlo al mayor.
    
    Si un índice no existe en uno de los textos, se genera un chunk vacío para ese texto, pero
    el número de chunks debe ser el mismo.
    
    Si un índice no se encuentra en texto1 pero es el índice intermedio (no el último), debemos 
    manejarlo correctamente generando los chunks para texto2 y dejando espacios vacíos en texto1.
    
    Args:
        texto1 (str): Texto completo del documento 1 (sin breaklines).
        texto2 (str): Texto completo del documento 2 (sin breaklines).
        indices (list[str]): Lista de marcadores de índice en el orden de aparición.
        chunk_size (int): Tamaño (en número de palabras) de cada fragmento.
        overlap (int): Cantidad de palabras de solapamiento entre fragmentos.
    
    Returns:
        tuple[list[str], list[str], list[str]]: 
            - Lista de fragmentos de `texto1`.
            - Lista de fragmentos de `texto2`.
            - Lista de índices relacionados con cada chunk.
    """
    chunks_texto1 = []
    chunks_texto2 = []
    chunks_indices = []  # Lista que asociará los chunks con los índices correspondientes
    
    # Para cada índice en el listado (asumido en orden de aparición)
    for i, marker in enumerate(indices):
        # Encontrar la posición de este índice en cada texto.
        start1 = texto1.find(marker)
        start2 = texto2.find(marker)
        
        # Si un índice no existe en uno de los textos, colocamos un fragmento vacío para ese texto
        if start1 == -1:
            segment1 = ""
            seg_chunks1 = [segment1]  # El chunk vacío para texto1
        else:
            # Definir el final del segmento: desde este marcador hasta el siguiente, o hasta el final.
            if i < len(indices) - 1:
                next_marker = indices[i + 1]
                end1 = texto1.find(next_marker, start1)
                if end1 == -1:
                    end1 = len(texto1)
            else:
                end1 = len(texto1)
            
            segment1 = texto1[start1:end1].strip()
            
            # Si el segmento tiene menos de 200 caracteres, lo dejamos en un único chunk.
            if len(segment1) < 200:
                seg_chunks1 = [segment1]
            else:
                words1 = segment1.split()
                n_chunks1 = math.ceil((len(words1) - overlap) / (chunk_size - overlap))
                seg_chunks1 = chunk_segment(words1, n_chunks1, chunk_size, overlap)
        
        # Lo mismo para texto2
        if start2 == -1:
            segment2 = ""
            seg_chunks2 = [segment2]  # El chunk vacío para texto2
        else:
            # Definir el final del segmento: desde este marcador hasta el siguiente, o hasta el final.
            if i < len(indices) - 1:
                next_marker = indices[i + 1]
                end2 = texto2.find(next_marker, start2)
                if end2 == -1:
                    end2 = len(texto2)
            else:
                end2 = len(texto2)
            
            segment2 = texto2[start2:end2].strip()
            
            # Si el segmento tiene menos de 200 caracteres, lo dejamos en un único chunk.
            if len(segment2) < 200:
                seg_chunks2 = [segment2]
            else:
                words2 = segment2.split()
                n_chunks2 = math.ceil((len(words2) - overlap) / (chunk_size - overlap))
                seg_chunks2 = chunk_segment(words2, n_chunks2, chunk_size, overlap)
        
        # Si un índice no existe en texto1 y no es el último índice, generamos los chunks de texto2
        # y añadimos vacíos en los de texto1
        if start1 == -1 and i < len(indices) - 1:
            common_chunks = len(seg_chunks2)
            seg_chunks1 = [""] * common_chunks
        
        # Forzar que el número de chunks para este índice sea el mismo en ambos textos.
        common_chunks = max(len(seg_chunks1), len(seg_chunks2))
        if len(seg_chunks1) != common_chunks:
            if len(segment1) < 200:
                seg_chunks1 = [segment1] * common_chunks
            else:
                seg_chunks1 = chunk_segment(segment1.split(), common_chunks, chunk_size, overlap)
        if len(seg_chunks2) != common_chunks:
            if len(segment2) < 200:
                seg_chunks2 = [segment2] * common_chunks
            else:
                seg_chunks2 = chunk_segment(segment2.split(), common_chunks, chunk_size, overlap)
        
        # Agregar los chunks de este índice a las listas finales.
        chunks_texto1.extend(seg_chunks1)
        chunks_texto2.extend(seg_chunks2)
        
        # Agregar el índice correspondiente para cada chunk
        chunks_indices.extend([marker] * common_chunks)
    
    return chunks_texto1, chunks_texto2, chunks_indices
