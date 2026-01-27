import chromadb

client = chromadb.Client()

collection = client.create_collection(name="test")

collection.add(
    documents=[
        "cricket is a sport.",
        "the sky is blue."
    ],
    ids=["1", "2"]
)

results = collection.query(
    query_texts=["sport"],
    n_results=2
)

print(results)
