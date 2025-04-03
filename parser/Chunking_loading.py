import difflib
from LLM import call_differences, bedrock_runtime
from psycopg2.extras import execute_values
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from db.connection import model

def retrieve_knn_difference(conn, list_indexes, query_text, k=5):
    """
    Dado un query, se obtiene su embedding y se recuperan los K chunks más similares usando la busqueda de vecinos.
    """
    list_indexes = str(list_indexes).replace("[","").replace("]","")
    cur = conn.cursor()
    query_embedding = model.encode(query_text).tolist()
    # La consulta utiliza el operador <-> (distancia euclidiana o coseno, según la configuración de PGVector)
    if list_indexes == '':
        sql = """
        SELECT indexes, text_diferences, 
               embedding <-> %s::vector AS distance
        FROM differences
            ORDER BY distance ASC
            LIMIT %s;
        """
        cur.execute(sql, (query_embedding, k))
        results = cur.fetchall()
        cur.close()
    else:
        sql = """
        SELECT similarity(indexes::text, %s::text) AS text_similarity, text_diferences, 
            embedding <-> %s::vector AS distance
        FROM differences
            WHERE similarity(indexes::text, %s::text) > 0
            ORDER BY distance ASC
            LIMIT %s;
        """

        cur.execute(sql, (list_indexes, query_embedding, list_indexes, k))
        results = cur.fetchall()
        cur.close()
    return results

def retrieve_knn_QA(conn, query_text, list_indexes, k=5):
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
        ORDER BY distance ASC
        LIMIT %s;
    """

    cur.execute(sql, (query_embedding, list_indexes, list_indexes, k))
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

def split_into_sentences(text: str) -> list[str]:
    """
    Splits a text into sentences. This is a simple splitter that assumes sentences end with 
    '.', '!' or '?' followed by whitespace.
    """
    # The regex looks for punctuation that likely terminates a sentence.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove any empty sentences and strip extra spaces.
    return [s.strip() for s in sentences if s.strip()]

def chunk_text_indexes_differences(texto1: str, texto2: str, indices: list[str]) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """
    For each index (section marker) in the given order, extracts the corresponding segments
    from texto1 and texto2 (from the marker up to the next marker or end of text). Then, splits
    each segment into sentences and computes the differences:
      - For text1: the sentences that do NOT appear in text2.
      - For text2: the sentences that do NOT appear in text1.
    
    If an index is not found in one of the texts, its segment is treated as empty.
    
    Args:
        texto1 (str): Full text of document 1 (assumed to be a single string without extra breaklines).
        texto2 (str): Full text of document 2 (same assumption).
        indices (list[str]): List of section markers (indices) in the order of appearance.
    
    Returns:
        tuple:
          - list[str]: The markers that were processed.
          - list[list[str]]: A list of lists; each sublist contains the sentences from texto1 for that marker
                             that do not appear in the corresponding segment of texto2.
          - list[list[str]]: Similarly, a list of lists for texto2 differences.
    """
    markers = []
    differences = []

    
    
    for i, marker in enumerate(indices):
        # Find segment boundaries in texto1
        start1 = texto1.find(marker)
        if start1 == -1:
            segment1 = ""
        else:
            markers.append(marker)
            if i < len(indices) - 1:
                next_marker = indices[i+1]
                end1 = texto1.find(next_marker, start1)
                if end1 == -1:
                    end1 = len(texto1)
            else:
                end1 = len(texto1)
            segment1 = texto1[start1:end1].strip()
        
        # Find segment boundaries in texto2
        start2 = texto2.find(marker)
        if start2 == -1:
            segment2 = ""
        else:
            if i < len(indices) - 1:
                next_marker = indices[i+1]
                end2 = texto2.find(next_marker, start2)
                if end2 == -1:
                    end2 = len(texto2)
            else:
                end2 = len(texto2)
            segment2 = texto2[start2:end2].strip()
        
        # Split the segments into sentences.
        
        prompt = (
        f"Utilizando el siguiente contexto responde la pregunta:\n\n"
        f"Context:\nTexto 1: {segment1} Texto 2: {segment2}\n\n"
        f"Question: ¿Cuales son las diferencias entre los Textos?\n"
        f"Answer:"
        )
        # Call the LLM to get the differences
        response = call_differences(bedrock_runtime, prompt, '¿Cuales son las diferencias entre los Textos?')
        difference = response['content'][0]['text'].strip()
        differences.append(difference)
    
    return markers, differences