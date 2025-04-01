import re
import difflib

def split_into_sentences(text: str) -> list[str]:
    """
    Splits a text into sentences. This is a simple splitter that assumes sentences end with 
    '.', '!' or '?' followed by whitespace.
    """
    # The regex looks for punctuation that likely terminates a sentence.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove any empty sentences and strip extra spaces.
    return [s.strip() for s in sentences if s.strip()]

def chunk_text_indexes_differences(texto1: str, texto2: str, indices: list[str]) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """
    For each index (section marker) in the given order, extracts the corresponding segments
    from texto1 and texto2 (from the marker up to the next marker or end of text). Then, splits
    each segment into sentences and computes the differences:
      - For text1: the sentences that do NOT appear in text2.
      - For text2: the sentences that do NOT appear in text1.
    
    If an index is not found in one of the texts, its segment is treated as empty.
    
    Args:
        texto1 (str): Full text of document 1 (assumed to be a single string without extra breaklines).
        texto2 (str): Full text of document 2 (same assumption).
        indices (list[str]): List of section markers (indices) in the order of appearance.
    
    Returns:
        tuple:
          - list[str]: The markers that were processed.
          - list[list[str]]: A list of lists; each sublist contains the sentences from texto1 for that marker
                             that do not appear in the corresponding segment of texto2.
          - list[list[str]]: Similarly, a list of lists for texto2 differences.
    """
    markers = []
    diffs_text1 = []
    diffs_text2 = []
    
    for i, marker in enumerate(indices):
        # Find segment boundaries in texto1
        start1 = texto1.find(marker)
        if start1 == -1:
            segment1 = ""
        else:
            if i < len(indices) - 1:
                next_marker = indices[i+1]
                end1 = texto1.find(next_marker, start1)
                if end1 == -1:
                    end1 = len(texto1)
            else:
                end1 = len(texto1)
            segment1 = texto1[start1:end1].strip()
        
        # Find segment boundaries in texto2
        start2 = texto2.find(marker)
        if start2 == -1:
            segment2 = ""
        else:
            if i < len(indices) - 1:
                next_marker = indices[i+1]
                end2 = texto2.find(next_marker, start2)
                if end2 == -1:
                    end2 = len(texto2)
            else:
                end2 = len(texto2)
            segment2 = texto2[start2:end2].strip()
        
        # Split the segments into sentences.
        sentences1 = split_into_sentences(segment1)
        sentences2 = split_into_sentences(segment2)
        
        # Use difflib.ndiff to compute the differences.
        # Lines starting with '- ' indicate sentences in text1 that are not in text2.
        # Lines starting with '+ ' indicate sentences in text2 that are not in text1.
        diff = list(difflib.ndiff(sentences1, sentences2))
        diff_sentences1 = [line[2:] for line in diff if line.startswith("- ")]
        diff_sentences2 = [line[2:] for line in diff if line.startswith("+ ")]
        
        markers.append(marker)
        diffs_text1.append(diff_sentences1)
        diffs_text2.append(diff_sentences2)
    
    return markers, diffs_text1, diffs_text2

# Example usage:
if __name__ == "__main__":
    # Sample texts with indices (for demonstration purposes)
    texto1 = (
        "ÍNDICE. I.TERMINOS DE REFERENCIA. "
        "1.DENOMINACIÓN DE LA CONTRATACIÓN. Some sentence unique to text1. "
        "2.FINALIDAD PÚBLICA. Common sentence. Another unique sentence in text1. "
        "3.ANTECEDENTES. Common sentence. End of section."
    )
    texto2 = (
        "ÍNDICE. I.TERMINOS DE REFERENCIA. "
        "1.DENOMINACIÓN DE LA CONTRATACIÓN. Common sentence. "
        "2.FINALIDAD PÚBLICA. Common sentence. A different unique sentence in text2. "
        "3.ANTECEDENTES. Common sentence. End of section."
    )
    
    indices = [
        "ÍNDICE. I.TERMINOS DE REFERENCIA.",
        "1.DENOMINACIÓN DE LA CONTRATACIÓN.",
        "2.FINALIDAD PÚBLICA.",
        "3.ANTECEDENTES."
    ]
    
    markers, diffs_t1, diffs_t2 = chunk_text_indexes_differences(texto1, texto2, indices)
    
    print(diffs_t1)