from loader import load_file
from vectorstore import show_relevant_score, build_vectorstore, build_retriever
from chain import build_chain
from splitter import splitter
import os

def load_multiple_documents(file_paths):
    all_documents = []
    for path in file_paths:
        documents = load_file(path)
        chunks = splitter(documents, chunk_size=1000, chunk_overlap=200)
        all_documents.extend(chunks)
        print(f"  '{path}' → {len(chunks)} chunks")
    return all_documents

def display_sources(context_data):
    print("\n[Sources used]")
    seen = set()
    for doc in context_data:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"{source} (page {page})" if page != "" else source
        if label not in seen:
            print(f"- {label}")
            seen.add(label)

def main():
    print("\n--- Basic RAG Pipeline ---\n")

    # Support Multiple Files
    print("Enter file paths one by one. Press Enter with no input when done.")
    file_paths = []
    while True:
        path = input("File path: ").strip()
        # If path doesn't exists then break out
        if not path:
            # If file_path is empty then skip that iteration
            if not file_paths:
                print("Please enter at least one file path.")
                continue
            break
        # If file path doesn't exist then inform the user about it and skip the iteration
        if not os.path.isfile(path):
            print(f"File not found: '{path}'. Please try again.")
            continue
        # After all the checks, add the file path to the list if it exists
        file_paths.append(path)

    # Load and Splitting all documents
    print("\nLoading and splitting documents...")
    all_chunks = load_multiple_documents(file_paths)
    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    # Build Vector Store and Retriever
    vectorstore = build_vectorstore(all_chunks)
    retriever = build_retriever(vectorstore, k=3)

    # Build RAG Chain
    rag_chain = build_chain(retriever)

    # Interactive Query Loop
    print("\nReady! Ask questions about your documents.")
    print("Commands: 'sources' to see all loaded files | 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if question.lower() == "sources":
            display_sources(all_chunks)
            continue
        if not question:
            print("Please enter a question or command.")
            continue

        # Show relevant chunks and their scores before answering
        show_relevant_score(vectorstore, question, k=3)

        # Get the answer from the RAG chain
        answer = rag_chain.invoke({"input": question})

        # Display the answer
        print(f"\nAnswer: {answer['answer']}\n")

        # Display sources used in the answer
        display_sources(answer['context'])
        print()

if __name__ == "__main__":
    main()
