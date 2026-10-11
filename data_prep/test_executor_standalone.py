import sys
sys.path.insert(0, 'd:/vscode/text_to_gql/data_prep')
from load_graphs import load_graph
from query_executor import execute_cypher_query
import json

G = load_graph('flight_accident')
print(f'Flight accident loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

with open('d:/Datasets/Cyberbench/test.json', encoding='utf-8') as f:
    test_qas = [q for q in json.load(f) if q['graph'] == 'flight_accident']

with open('d:/vscode/text_to_gql/data/maps/flight_accident_k4.json', encoding='utf-8') as f:
    pmap = json.load(f)

cross = 0
for q in test_qas[:10]:
    nodes = execute_cypher_query(G, q['gold_cypher'])
    pids = set(pmap[n] for n in nodes if n in pmap)
    if len(pids) > 1:
        cross += 1
    print(f"Touched: {len(nodes)}, Partitions: {pids}")

print(f"Cross in 10 samples: {cross}/10")

