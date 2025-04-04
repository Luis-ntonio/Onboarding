from bert_score import score
import sqlite3
from db.connection import create_conn

conn = create_conn()
cursor = conn.cursor()

# Query to fetch gpt_answer and rag_answer from the comparison table
query = "SELECT gpt_answer, rag_answer FROM comparison"
cursor.execute(query)
rows = cursor.fetchall()

# Compare answers using BERTScore
for idx, (gpt_answer, rag_answer) in enumerate(rows):
    if gpt_answer and rag_answer:  # Ensure both answers are not None
        P, R, F1 = score([gpt_answer], [rag_answer], lang="en", verbose=False)
        print(f"Row {idx + 1}:")
        print(f"  Precision: {P.item():.4f}")
        print(f"  Recall: {R.item():.4f}")
        print(f"  F1 Score: {F1.item():.4f}")
        print()

# Close the database connection
conn.close()