from pymilvus import connections, Collection, utility

class CME_Milvus:
    def __init__(self, collection_name):
        connections.connect(alias="default", host='localhost', port='19530')
        self.collection_name = collection_name
        if not utility.has_collection(self.collection_name):
            self._create_collection()
        self.collection = Collection(self.collection_name)

    def _create_collection(self):
        from pymilvus import FieldSchema, CollectionSchema, DataType
        fields = [
            FieldSchema(name="symbol_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=158)
        ]
        schema = CollectionSchema(fields, "CME symbol embeddings")
        self.collection = Collection(name=self.collection_name, schema=schema)

    def insert_embedding(self, symbol_id, embedding):
        data = [
            [symbol_id],
            [embedding]
        ]
        self.collection.insert(data)
        self.collection.flush()

    def semantic_search(self, query_embedding, top_k=10):
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search([query_embedding], "embedding", search_params, top_k)
        return results