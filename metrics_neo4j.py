from src.cme_neo4j import CME_Neo4j

class Neo4jMetrics:
    def __init__(self, uri, user, password):
        self.neo4j = CME_Neo4j(uri, user, password)

    def detect_cycle(self, start_symbol_id, max_depth=10):
        with self.neo4j.driver.session() as session:
            result = session.execute_read(self._detect_cycle, start_symbol_id, max_depth)
            return result

    @staticmethod
    def _detect_cycle(tx, start_symbol_id, max_depth):
        result = tx.run("MATCH path=(start:Symbol {id: $id})-[*1..$depth]->(start) "
                        "RETURN path", id=start_symbol_id, depth=max_depth)
        return [record for record in result]