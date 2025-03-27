import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuración del modelo para obtener embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Este modelo genera vectores de dimensión 384

def cargar_chunks_desde_archivo(conn, archivo_chunks):
    """
    Lee los chunks desde un archivo y los inserta en la base de datos.
    """
    with open(archivo_chunks, "r") as f:
        chunks = [line.strip() for line in f.readlines()]
    insert_chunks(conn, chunks)

def create_table(conn, embedding_dim=384):
    """
    Crea la tabla 'chunks' en PostgreSQL utilizando la extension PGVector.
    Se asume que la extensión 'vector' esta instalada en la base de datos.
    """
    cur = conn.cursor()
    # Asegurarse de que la extensión PGVector esté instalada
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Crear la tabla para almacenar los chunks y sus embeddings
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding VECTOR({embedding_dim})
        );
    """)
    conn.commit()
    cur.close()

def insert_chunks(conn, chunks):
    """
    Para cada chunk, se calcula su embedding y se inserta junto con el texto en la tabla.
    """
    cur = conn.cursor()
    data = []
    for chunk in chunks:
        # Obtener el embedding como lista de floats
        embedding = model.encode(chunk).tolist()
        data.append((chunk, embedding))
    query = "INSERT INTO chunks (text, embedding) VALUES %s"
    execute_values(cur, query, data)
    conn.commit()
    cur.close()

def insert_chunks_db(conn, chunks):
    """
    Para cada chunk, se calcula su embedding y se inserta junto con el texto en la tabla.
    'chunks' debe ser una lista de strings.
    """
    cur = conn.cursor()
    data = []
    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        data.append((chunk, embedding))

    query = "INSERT INTO chunks (text, embedding) VALUES %s"
    execute_values(cur, query, data)
    conn.commit()
    cur.close()

def retrieve_knn(conn, query_text, k=5):
    """
    Dado un query, se obtiene su embedding y se recuperan los K chunks más similares usando la busqueda de vecinos.
    """
    cur = conn.cursor()
    query_embedding = model.encode(query_text).tolist()
    # La consulta utiliza el operador <-> (distancia euclidiana o coseno, según la configuración de PGVector)
    sql = """
        SELECT id, text, embedding <-> %s::vector AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s;
    """

    cur.execute(sql, (query_embedding, k))
    results = cur.fetchall()
    cur.close()
    return results

def create_conn():
    conn = psycopg2.connect(
        dbname="amber",
        user="postgres",
        password="1234",
        host="localhost"
    )
    return conn

if __name__ == "__main__":
    conn = create_conn()
    
    create_table(conn)
    

    archivo_chunks1 = "./data/documento_uniforme2.txt_1_chunks.txt"
    archivo_chunks2 = "./data/documento_uniforme2.txt_2_chunks.txt"
    
    cargar_chunks_desde_archivo(conn, archivo_chunks1)
    cargar_chunks_desde_archivo(conn, archivo_chunks2)

    query_text = "informacion relevante sobre el tema"
    results = retrieve_knn(conn, query_text, k=3)
    for row in results:
        print("ID:", row[0], "Distancia:", row[2], "Texto:", row[1])
    
    conn.close()
