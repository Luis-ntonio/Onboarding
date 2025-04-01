from psycopg2.extensions import register_adapter, AsIs
from sentence_transformers import SentenceTransformer
import numpy as np
from db.connection import model
from psycopg2.extras import execute_values

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64) 

def create_difference_table(conn, embedding_dim=384):
    """
    Crea la tabla 'chunks' en PostgreSQL utilizando la extension PGVector.
    Se asume que la extensión 'vector' esta instalada en la base de datos.

    Args:
        conn: Conexión a la base de datos.
        embedding_dim (int): Dimensión del embedding a almacenar.

    Returns:
        None
    """
    cur = conn.cursor()
    # Asegurarse de que la extensión PGVector esté instalada
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Crear similarity
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_similarity;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")


    # Crear la tabla para almacenar los chunks y sus embeddings
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS differences (
            id SERIAL PRIMARY KEY,
            indexes TEXT,
            text_diferences TEXT,
            embedding VECTOR({embedding_dim})
        );
    """)
    conn.commit()
    cur.close()

def insert_differences_chunks(conn, chunks1, indexes):
    """
    Para cada par de chunks, se calcula su embedding, la similitud, y se inserta junto con los textos en la tabla.
    
    Args:
        conn: Conexión a la base de datos.
        chunks1 (list): Lista de chunks del primer texto.
        chunks2 (list): Lista de chunks del segundo texto.
        indexes (list): Lista de índices correspondientes a los chunks.
    """
    cur = conn.cursor()
    data = []
    for i, chunk1 in enumerate(chunks1):
        # Calcular embeddings para ambos chunks
        embedding1 = model.encode(chunk1).tolist()

        # Preparar los datos para la inserción
        data.append((indexes[i], chunk1, embedding1))

    # Query para insertar en la tabla 'differences'
    query = f"""
        INSERT INTO differences (indexes, text_diferences, embedding)
        VALUES %s
    """
    execute_values(cur, query, data)
    conn.commit()
    cur.close()