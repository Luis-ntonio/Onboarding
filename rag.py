from Chunking_loading import retrieve_knn, create_conn
from First_call import claude_call, bedrock_runtime
from Parser_pdf2 import remove_connector_words, normalize_text, lemmatize_text 


def rag_call(query_text, conn, k=5):
    """
    Performs a RAG call by:
      - Retrieving the k most similar chunks from the database.
      - Constructing a prompt that includes these chunks as context.
      - Calling a generative model to produce an answer.
    """

    query_text_tmp = remove_connector_words(query_text)
    query_text_tmp = normalize_text(query_text_tmp)
    query_text_tmp = lemmatize_text(query_text_tmp)

    # Retrieve the k nearest chunks using the retrieval function defined earlier
    results = retrieve_knn(conn, query_text_tmp, k)
    
    # Concatenate the retrieved chunks to form the context
    context = "\n---\n".join([row[1] for row in results])
    
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

# Example usage:
# Assuming you already have a PostgreSQL connection 'conn'
query = "¿Cual es la finalidad publica de los documentos?"
conn = create_conn()
answer = rag_call(query, conn, k=5)
print("Answer:", answer)
