from retriever import retrieve

question = "What are the prerequisites for AI-101?"

results = retrieve(question, k=3)

for rank, chunk in enumerate(results, start=1):
    print(f"\nResult #{rank}")
    print("Source:", chunk["source"])
    print("Distance:", round(chunk["distance"], 4))
    print("Text:", chunk["text"])