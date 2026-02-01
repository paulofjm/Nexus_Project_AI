import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import faiss
import numpy as np
# from sentence_transformers import SentenceTransformer
from models.embedding import embed_text
from core.storage import faiss_index, metadata
from utils.logger import write_monitor_log as log


# Carregar FAISS e metadados
log("FAISS carregado.")
if not os.path.exists(faiss_index):
    raise FileNotFoundError(f"Índice FAISS não encontrado em {faiss_index}")

index = faiss.read_index(faiss_index)

with open(metadata, "r", encoding="utf-8") as f:
    metadata = json.load(f)

log(f"Índice com {len(metadata)} registros.")

# Carregar modelo de embeddings
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Função de busca
def search_activity(query, top_k=5, tag_filter=None):
    query_embedding = embed_text([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            result = metadata[idx]
            result['distance'] = float(dist)

            # Se tiver filtro por tag, só adiciona se uma das tags bater
            if tag_filter:
                tags = result.get("tags", [])
                if not any(tag_filter.lower() in tag.lower() for tag in tags):
                    continue

            results.append(result)
    
    return results

if __name__ == "__main__":
    while True:
        user_query = input("\n🔍 Digite sua busca (ou 'sair' para terminar): ")
        if user_query.lower() == "sair":
            break

        tag_filter = input("🔎 (Opcional) Filtrar por tag? (ex: python, vscode, api): ").strip()
        tag_filter = tag_filter if tag_filter else None

        resultados = search_activity(user_query, top_k=5, tag_filter=tag_filter)

        if not resultados:
            print("Nenhuma atividade relevante encontrada.")
            continue