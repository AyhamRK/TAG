from app.db.vector_store import VectorStore

store = VectorStore()
print("Total chunks:", store.count())
print("Indexed books:", store.get_indexed_books())
print()

results = store.similarity_search("knowledge economy", top_k=3)
for meta, score in results:
    print(f"[{score:.3f}] {meta['book_name']} p.{meta['page_number']}")
    print(f"  {meta['text'][:100]}")
    print()
