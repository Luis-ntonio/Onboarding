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
    prompt = (
        f"Estas encargado de analizar preguntas para extraer secciones, apartados o indices e indicar si existe uno o mas de uno.\n\n"
        f"Options: UNA | MAS DE UNA\n"
        f"Example: ¿Cuáles son las diferencias en los apartados 1. antecedentes y 2.1 Motivacion?\n"
        f"Template Answer: MAS DE UNA\n"
        f"Example: ¿Cuáles son las diferencias en el Indice 1?\n"
        f"Template Answer: UNA\n"
        f"Answer: "
    )
    response = claude_call(bedrock_runtime, prompt, query_text)
    # Extract and return the generated answer
    answer = response['content'][0]['text'].strip()
    if answer == "UNA":
        k = 1
        results = retrieve_knn_difference(conn, query_text_tmp, k)
        answer = "\n---\n".join([row[1] for row in results])
    else:
        results = retrieve_knn_difference(conn, query_text_tmp, k)
        answer = "\n---\n".join([row[1] for row in results])
        prompt = (
            f"Estas encargado de analizar y resumir texto sin perjudicar el contexto\n\n"
            f"Answer: "
        )
        response = claude_call(bedrock_runtime, prompt, answer)
        # Extract and return the generated answer
        answer = response['content'][0]['text'].strip()

    # Concatenate the retrieved chunks to form the context
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