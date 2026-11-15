
from retriever import retrieve
from prompt_builder import build_context
from generator import generate_answer

while True:

    question = input("\nAsk your document assistant: ")

    if question.lower().strip() == "exit":
        break

    # STEP 1: Retrieve relevant chunks
    chunks = retrieve(question, k=5)

    if not chunks:
        print("I couldn't find relevant information.")
        continue

    # STEP 2: Build organized context
    context, sources = build_context(chunks)

    # STEP 3: Generate an answer
    answer = generate_answer(question, context)

    # STEP 4: Display answer
    print("\nANSWER:")
    print(answer)

    # STEP 5: Display available source references
    print("\nAVAILABLE SOURCES:")

    for source in sources:
        print(
            f"[{source['id']}] {source['source']} "
            f"(chunk {source['chunk_index']})"
        )
