import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
    """Extrae el texto de un archivo PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def clean_and_uniformize_text(text):
    """Limpia y uniformiza el texto extraído."""
    # Eliminar espacios extra y líneas vacías
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    
    # Corregir enumeraciones (ejemplo básico)
    cleaned_text = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1.\2.\3', cleaned_text)
    
    # Aquí puedes agregar más reglas de limpieza y uniformización
    return cleaned_text

def parse_and_uniformize_pdfs(pdf1_path, pdf2_path, output_path):
    """Procesa y uniformiza dos PDFs."""
    # Extraer texto de ambos PDFs
    text1 = extract_text_from_pdf(pdf1_path)
    text2 = extract_text_from_pdf(pdf2_path)
    
    # Limpieza y uniformización
    uniform_text1 = clean_and_uniformize_text(text1)
    uniform_text2 = clean_and_uniformize_text(text2)
    
    # Combinar ambos textos
    combined_text = f"Documento 1:\n{uniform_text1}\n\nDocumento 2:\n{uniform_text2}"
    
    # Guardar el texto combinado en un archivo
    with open(output_path, "w") as output_file:
        output_file.write(combined_text)

# Ejemplo de uso
pdf1_path = "./data/tdr_v4.pdf"
pdf2_path = "./data/tdr_v6.pdf"
output_path = "./data/documento_uniforme.txt"

parse_and_uniformize_pdfs(pdf1_path, pdf2_path, output_path)