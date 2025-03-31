from parser.Chunking_loading import retrieve_knn, create_conn
from LLM import claude_call, bedrock_runtime
from parser.Parser_pdf2 import remove_connector_words, normalize_text 
from Questions import Querys

def rag_call(query_text, conn, k=5):
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
    results = retrieve_knn(conn, query_text_tmp, k)
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
    answer = rag_call(query, conn, k=5)
    print("Query:", query)
    print("Answer:", answer)
    print()