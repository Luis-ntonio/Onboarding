from parser.Parser_pdf2 import extraer_texto, eliminar_indice, remove_connector_words, remove_pagination_words
from parser.Chunking_loading import chunk_text, chunk_text_indexes
from db.embedding_db import create_embedding_table, insert_embedding_chunks
from db.difference_db import create_difference_table, insert_differences_chunks
from db.connection import create_conn


def parser_uniformizador(pdf_path1: str, pdf_path2: str, salida_base: str) -> None:
    """
    Procesa dos PDFs:
      - Extrae el texto.
      - Elimina el indice.
      - Renumera las secciones.
      - Elimina palabras conectivas.
      - Normaliza el texto (minusculas, eliminacion de tildes y signos de puntuacion).
      - Elimina palabras con formato "pagina1de2" o que contengan "ndem".
      - Aplica lematización: transforma verbos a infinitivo y pronombres plurales a singular (según mapeo).
      - Divide el contenido en fragmentos (chunks) para uso en sistemas RAG.
      
    Se generan archivos de salida para cada PDF:
      - Un archivo con el texto uniformizado.
      - Un archivo con los chunks, separados por una línea delimitadora.

    Args:
        pdf_path1 (str): Ruta al primer PDF.
        pdf_path2 (str): Ruta al segundo PDF.
        salida_base (str): Nombre base para los archivos de salida. Se anhade sufijo "_1" y "_2" para cada PDF
    
    Returns:
        None
    """
    name1 = pdf_path1.split("/")[-1].split(".")[0]
    name2 = pdf_path2.split("/")[-1].split(".")[0]
    print(f"Procesando PDFs: {name1} y {name2}")

    texto1, titulo1 = extraer_texto(pdf_path1, 1)
    texto2, titulo2 = extraer_texto(pdf_path2, 2)

    texto1, indexes1 = eliminar_indice(texto1, titulo1)
    texto2, indexes2 = eliminar_indice(texto2, titulo2)
    
    texto1 = remove_connector_words(texto1)
    texto2 = remove_connector_words(texto2)
    
    texto1 = remove_pagination_words(texto1)
    texto2 = remove_pagination_words(texto2)
    
    chunks1, indexes1_ = chunk_text(texto1, indexes1)
    chunks2, indexes2_ = chunk_text(texto2, indexes2)

    #get unique indexes merge
    indexes_diff = list(set(indexes1_).difference(set(indexes2_)))
    indexes_diff += list(set(indexes2_).difference(set(indexes1_)))
    indexes_diff = sorted(indexes_diff)

    chunk_diff1, chunk_diff2, indexes_diff = chunk_text_indexes(texto1, texto2, indexes_diff)
    
    with open(salida_base + "_1.txt", "w", encoding="utf-8") as f:
        f.write(texto1)
    with open(salida_base + "_2.txt", "w", encoding="utf-8") as f:
        f.write(texto2)
    
    with open(salida_base + "_1_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks1:
            f.write(chunk + "\n")
    with open(salida_base + "_2_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks2:
            f.write(chunk + "\n")
    
    conn = create_conn()
    create_embedding_table(conn)
    insert_embedding_chunks(conn, chunks1, indexes1_, name1)
    insert_embedding_chunks(conn, chunks2, indexes2_, name2)
    create_difference_table(conn)
    insert_differences_chunks(conn, chunk_diff1, indexes_diff, name1)
    conn.close()

    print("Proceso completado. Archivos guardados:")
    print("Texto uniformizado:", salida_base + "_1.txt", "y", salida_base + "_2.txt")
    print("Chunks para RAG:", salida_base + "_1_chunks.txt", "y", salida_base + "_2_chunks.txt")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Uso: python parser.py archivo1.pdf archivo2.pdf salida_base")
    else:
        parser_uniformizador(sys.argv[1], sys.argv[2], sys.argv[3])