from parser.Chunking_loading import retrieve_knn_difference, retrieve_knn_QA
from db.connection import create_conn
from LLM import claude_call, bedrock_runtime
from parser.Parser_pdf2 import remove_connector_words, normalize_text 
from Questions import Querys
from diferencias import diferencias
from db.comparison_db import create_comparison_table, insert_comparison
import ast

# Read indexes.txt file
with open('index.txt', 'r') as file:
    indexes = file.read().splitlines()
# convert it into array
indexes = [line.split(' ')[0] for line in indexes]

def get_indexes(query):
    prompt = (
        f"Estas encargado de analizar preguntas para extraer secciones, apartados o indices e indicar que secciones de las Options se estan pidiendo.\n\n"
        f"Options: {indexes}\n"
        f"Example: ¿Cuáles son las diferencias en los apartados 1. DENOMINACIÓN DE LA CONTRATACIÓN  y 2 FINALIDAD PÚBLICA ?\n"
        f"Template Answer: [1. DENOMINACION DE LA CONTRATACION, 2. FINALIDAD PUBLICA\n"
        f"Example: ¿Cuáles son las diferencias en el Indice 1?\n"
        f"Template Answer: [1. DENOMINACION DE LA CONTRATACION]\n"
        f"Example: ¿Cual es la motivacion del documento?\n"
        f"Template Answer: []\n"
        f"Example: ¿Las certificaciones requeridas para el proveedor de nube pública son las mismas en ambas versiones del documento?\n"
        f"Template Answer: []\n"
        f"Answer: "
    )
    response = claude_call(bedrock_runtime, prompt, query)
    # Extract and return the generated answer
    answer = response['content'][0]['text'].strip()
    # answer has to be only the part of the text similar as list []
    # Extract the list-like portion of the answer
    try:
        list_str = answer.split('[')[1].split(']')[0]
        #convert it into a list
        answer = list_str.split(',')
    except:
        answer = []

    return answer

def rag_call_differences(query_text, conn, list_indexes, k=5):
    """
    Performs a RAG call by:
      - Retrieving the k most similar chunks from the database.
      - Constructing a prompt that includes these chunks as context.
      - Calling a generative model to produce an answer.
    
    Args:
        query_text (str): The question or query text.
        conn: A connection to the database.
        k (int): The number of chunks to retrieve.
    
    Returns:
        str: The generated answer.
    """

    query_text_tmp = remove_connector_words(query_text)
    query_text_tmp = normalize_text(query_text_tmp)

    # Retrieve the k nearest chunks using the retrieval function defined earlier
    if len(list_indexes) == 1:
        k = 1
        if list_indexes[0] == '':
            list_indexes = ''
            k = 5
        results = retrieve_knn_difference(conn,list_indexes, query_text_tmp, k=k)
        answer = "\n---\n".join([row[1] for row in results])
    else:
        if list_indexes == []:
            results = retrieve_knn_difference(conn, list_indexes = '', query_text=query_text_tmp, k=k)
        else:
            results = retrieve_knn_difference(conn, list_indexes, query_text_tmp, k=k)
        answer = "\n---\n".join([row[1] for row in results])
    prompt = (
        f"Estas encargado de analizar, resumir texto sin perjudicar el Context y responder a la pregunta solicitada\n\n"
        f"Context:\n{answer}\n\n"
        f"Answer: "
    )
    response = claude_call(bedrock_runtime, prompt, query_text)
    # Extract and return the generated answer
    answer = response['content'][0]['text'].strip()
    # Concatenate the retrieved chunks to form the context
    return answer

def rag_call_QA(query_text, conn, list_indexes,  k=5):
    """
    Performs a RAG call by:
      - Retrieving the k most similar chunks from the database.
      - Constructing a prompt that includes these chunks as context.
      - Calling a generative model to produce an answer.
    
    Args:
        query_text (str): The question or query text.
        conn: A connection to the database.
        k (int): The number of chunks to retrieve.
    
    Returns:
        str: The generated answer.
    """

    query_text_tmp = remove_connector_words(query_text)
    query_text_tmp = normalize_text(query_text_tmp)

    # Retrieve the k nearest chunks using the retrieval function defined earlier
    results = retrieve_knn_QA(conn, query_text_tmp, list_indexes, k)
    # Concatenate the retrieved chunks to form the context
    context = "\n---\n".join([row[2] for row in results])
    # Build a prompt that provides context and then asks the query
    prompt = (
        f"Eres un analizador de Textos. Utilizando el siguiente contexto responde la pregunta:\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )

    response = claude_call(bedrock_runtime, prompt, query_text)
    
    # Extract and return the generated answer
    answer = response['content'][0]['text'].strip()
    return answer

conn = create_conn()
# Create the database table if it doesn't exist
create_comparison_table(conn)

for query in Querys:
    prompt = (
        f"Estas encargado de analizar preguntas para conocer si se busca comparar documentos o no\n\n"
        f"Options: NO | YES\n"
        f"Example: ¿Cuáles son las diferencias en los apartados 1. antecedentes y 2.1 Motivacion?\n"
        f"Template Answer: YES\n"
        f"Example: ¿Cuál es el objetivo del documento?\n"
        f"Template Answer: NO\n"
        f"Answer: "
    )
    response = claude_call(bedrock_runtime, prompt, query)
    # Extract and return the generated answer
    answer = response['content'][0]['text'].strip()
    list_indexes = get_indexes(query)
    # Check if the answer indicates a comparison

    if "YES" in answer:
        answer = rag_call_differences(query, conn, list_indexes)
    else:
        answer = rag_call_QA(query, conn, list_indexes)

    # Insert the comparison into the database
    insert_comparison(conn, query, answer)

conn.close()