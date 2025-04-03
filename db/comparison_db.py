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

def create_comparison_table(conn, embedding_dim=384):
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
        CREATE TABLE IF NOT EXISTS comparison (
            id SERIAL PRIMARY KEY,
            question TEXT,
            rag_answer TEXT,
            gpt_answer TEXT,
            bert_metrics TEXT
        );
    """)
    conn.commit()
    cur.close()

def insert_comparison(conn, question, rag_answer, gpt_answer="", bert_metrics=""):
    """
    Inserta una comparación en la tabla 'comparison'.

    Args:
        conn: Conexión a la base de datos.
        question (str): Pregunta realizada.
        rag_answer (str): Respuesta generada por RAG.
        gpt_answer (str): Respuesta generada por GPT.
        bert_metrics (str): Métricas de similitud de BERT.

    Returns:
        None
    """
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO comparison (question, rag_answer, gpt_answer, bert_metrics)
        VALUES (%s, %s, %s, %s);
    """, (question, rag_answer, gpt_answer, bert_metrics))
    conn.commit()
    cur.close()
