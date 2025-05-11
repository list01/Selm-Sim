from neo4j import GraphDatabase

class CME_Neo4j:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_symbol_node(self, symbol_id, symbol_text):
        with self.driver.session() as session:
            session.execute_write(self._create_and_return_symbol, symbol_id, symbol_text)

    @staticmethod
    def _create_and_return_symbol(tx, symbol_id, symbol_text):
        result = tx.run("CREATE (a:Symbol {id: $id, text: $text}) "
                        "RETURN a.id AS id, a.text AS text", id=symbol_id, text=symbol_text)
        return result.single()

    def topological_query(self, start_symbol_id, depth):
        with self.driver.session() as session:
            result = session.execute_read(self._topological_query, start_symbol_id, depth)
            return result

    @staticmethod
    def _topological_query(tx, start_symbol_id, depth):
        result = tx.run("MATCH (start:Symbol {id: $id})-[*1..$depth]-(connected) "
                        "RETURN connected.id AS id, connected.text AS text", id=start_symbol_id, depth=depth)
        return [record for record in result]