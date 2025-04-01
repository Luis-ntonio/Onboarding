def has_differences(chunks_texto1: list[str], chunks_texto2: list[str]) -> list[bool]:
    """
    Compares the corresponding chunks from two texts and checks if there are differences.
    Returns True if there are differences, otherwise False.

    Args:
        chunks_texto1 (list[str]): List of chunks from text1.
        chunks_texto2 (list[str]): List of chunks from text2.
    
    Returns:
        list[bool]: List where each element indicates whether there are differences between corresponding chunks.
    """
    differences = []
    
    # Iterate through both lists of chunks (for text1 and text2)
    for chunk1, chunk2 in zip(chunks_texto1, chunks_texto2):
        # Compare the two chunks directly (using exact equality)
        if chunk1 != chunk2:
            differences.append(True)
        else:
            differences.append(False)
    
    return differences

# Example of how to use it:
chunks_texto1 = ["This is a sample chunk of text.", "Another chunk here.", "Final chunk."]
chunks_texto2 = ["This is a sample chunk of text.", "Another chunk here with some differences.", "Final chunk."]

# Get whether chunks have differences or not
differences = has_differences(chunks_texto1, chunks_texto2)

# Print the result
for i, diff in enumerate(differences):
    print(f"Chunk {i+1} has differences: {diff}")
