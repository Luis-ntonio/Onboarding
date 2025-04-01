from parser.Chunking_loading import retrieve_knn_difference, retrieve_knn_QA
from db.connection import create_conn
from LLM import claude_call, bedrock_runtime
from parser.Parser_pdf2 import remove_connector_words, normalize_text 
from Questions import Querys
from diferencias import diferencias



def rag_call_differences(query_text, conn, k=5):
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
    results = retrieve_knn_difference(conn, query_text_tmp, k)
    # Concatenate the retrieved chunks to form the context
    texto_1 = " ".join([row[2] for row in results])
    texto_2 = " ".join([row[3] for row in results])
    print("Texto 1:", texto_1)
    print("Texto 2:", texto_2)
    context = "Estos son los textos de solicitados para comparar \n Texto 1:\n" + texto_1 + "\n\nTexto 2:\n" + texto_2
    """for i in results:
        print(i[0], i[3], i[4], i[5])"""
    # Build a prompt that provides context and then asks the query
    prompt = (
        f"Utilizando el siguiente contexto responde la pregunta:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: ¿Cuales son las diferencias entre los Texto 1 y Texto 2?\n"
        f"Answer:"
    )

    response = claude_call(bedrock_runtime, prompt, query_text)
    
    # Extract and return the generated answer
    answer = response['content'][0]['text'].strip()
    return answer

def rag_call_QA(query_text, conn, k=5):
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
    results = retrieve_knn_QA(conn, query_text_tmp, k)
    # Concatenate the retrieved chunks to form the context
    context = "\n---\n".join([row[2] for row in results])
    # Build a prompt that provides context and then asks the query
    prompt = (
        f"Utilizando el siguiente contexto responde la pregunta:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query_text}\n"
        f"Answer:"
    )

    response = claude_call(bedrock_runtime, prompt, query_text)
    
    # Extract and return the generated answer
    answer = response['content'][0]['text'].strip()
    return answer

conn = create_conn()

for query in Querys:
    if any(dif in query for dif in diferencias):
        answer = "YES"
    else:
        answer = "NO"
    if "YES" in answer:
        answer = rag_call_differences(query, conn)
    else:
        answer = rag_call_QA(query, conn)

    print("Query:", query)
    print("Answer:", answer)
    print()