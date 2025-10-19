import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from datetime import datetime
from collections import Counter
import hashlib
import faiss
from langchain_community.vectorstores import FAISS
import numpy as np
from models.embedding import embed_text
from core.storage import FAISS_INDEX_PATH, METADATA_PATH, LOG_DIR, BASE_DIR, TEMP_DIR
from utils.logger import write_monitor_log as log
# Caminhos das memórias

# RAW_BUFFER_PATH = "data/temp/activity_raw_buffer.json"
RAW_BUFFER_PATH = os.path.join(TEMP_DIR, "activity_raw_buffer.json")
# COMPILED_BLOCKS_PATH = "data/logs/activity_compiled_blocks.json"
COMPILED_BLOCKS_PATH = os.path.join(LOG_DIR, "activity_compiled_blocks.json")
# LONG_TERM_PATH = "data/logs/activity_hourly_context.json"
LONG_TERM_PATH = os.path.join(LOG_DIR, "activity_hourly_context.json")

# === Carregadores utilitários ===
def carregar_json(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# === Inicializa FAISS e Metadata vazios se não existirem ===
def init_faiss_if_missing():
    if not os.path.exists(FAISS_INDEX_PATH):
        log("Inicializando FAISS vazio")
        index = faiss.IndexFlatL2(384)
        faiss.write_index(index, FAISS_INDEX_PATH)
    if not os.path.exists(METADATA_PATH):
        log("Inicializando metadata vazio")
        salvar_json(METADATA_PATH, [])



# === Memória de Curto Prazo ===
def salvar_em_buffer(record):
    buffer = carregar_json(RAW_BUFFER_PATH)
    buffer.append(record)
    if len(buffer) > 20:
        buffer = buffer[-20:]
    salvar_json(RAW_BUFFER_PATH, buffer)

    # Também indexa no FAISS para que o modelo tenha contexto imediato
    # texto_base = f"{record['active_window']} {record['extracted_text']}"
    texto_base = f"{record['active_window']} {record.get('context_summary', '')}"
    embedding = embed_text([texto_base])[0]

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(384)
    index.add(np.array([embedding], dtype=np.float32))
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Salva no metadata (temporário) como "curto_prazo"
    metadata = carregar_json(METADATA_PATH)
    metadata.append(record)
    salvar_json(METADATA_PATH, metadata)

# === Compactar bloco de 20 para memória de médio prazo ===
def compactar_bloco_de_20():
    buffer = carregar_json(RAW_BUFFER_PATH)
    if len(buffer) < 20:
        return False

    bloco = buffer[:20]
    t0 = datetime.fromisoformat(bloco[0]["timestamp"])
    tN = datetime.fromisoformat(bloco[-1]["timestamp"])

    janelas = [r["active_window"] for r in bloco]
    tags = sum((r.get("tags", []) for r in bloco), [])
    janela_mais_comum = Counter(janelas).most_common(1)[0][0]
    principais_tags = [t for t, _ in Counter(tags).most_common(5)]
    blocos_horarios = [
        f"{bloco[i]['timestamp'][11:16]}-{bloco[i+4]['timestamp'][11:16]}: {bloco[i]['active_window']}"
        for i in range(0, 20, 5)
    ]

    resumo = {
        "nivel": "medio",
        "timestamp_inicio": t0.isoformat(),
        "timestamp_fim": tN.isoformat(),
        "hash": hashlib.sha256(f"{t0}{tN}".encode()).hexdigest(),
        "resumo_compilado": True,
        "periodo": f"{t0.strftime('%H:%M')} às {tN.strftime('%H:%M')}",
        "janela_mais_frequente": janela_mais_comum,
        "principais_tags": principais_tags,
        "context_summary": "Atividades principais:\n" + "\n".join(blocos_horarios),
        "quantidade_registros": 20
    }

    # Salvar em memória de médio prazo
    blocos = carregar_json(COMPILED_BLOCKS_PATH)
    blocos.append(resumo)
    
    if len(blocos) > 36:
        blocos = blocos[-16:]  # mantém apenas os últimos 16
    salvar_json(COMPILED_BLOCKS_PATH, blocos)

    # Salvar no FAISS (sincronizado com metadata)
    embedding = embed_text([resumo["context_summary"]])[0]
    index = faiss.IndexFlatL2(384)
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    index.add(np.array([embedding], dtype=np.float32))
    faiss.write_index(index, FAISS_INDEX_PATH)

    metadata = carregar_json(METADATA_PATH)
    metadata.append(resumo)
    if len(metadata) > len(blocos):  # sincroniza com blocos válidos
        metadata = metadata[-len(blocos):]
        index = faiss.IndexFlatL2(384)
    for b in metadata:
        if "context_summary" in b:
            vec = embed_text([b["context_summary"]])[0]
            index.add(np.array([vec], dtype=np.float32))
        faiss.write_index(index, FAISS_INDEX_PATH)
    salvar_json(METADATA_PATH, metadata)

    # Atualiza buffer mantendo os últimos 10 registros
    salvar_json(RAW_BUFFER_PATH, buffer[-10:])
    return True


# === Consolidar blocos de 36 em memória de longo prazo ===
def consolidar_blocos_medios():
    blocos = carregar_json(COMPILED_BLOCKS_PATH)

    # Só consolida se houver pelo menos 36 blocos
    while len(blocos) >= 36:
        blocos_para_consolidar = blocos[:36]

        superbloco = {
            "nivel": "longo",
            "timestamp_gerado": datetime.now().isoformat(),
            "blocos_compilados": 36,
            "periodo": f"{blocos_para_consolidar[0]['timestamp_inicio'][11:16]} às {blocos_para_consolidar[-1]['timestamp_fim'][11:16]}",
            "resumos": [b["context_summary"] for b in blocos_para_consolidar],
            "tags": list(set(t for b in blocos_para_consolidar for t in b["principais_tags"]))
        }

        # Salva no JSON de longo prazo
        long_term = carregar_json(LONG_TERM_PATH)
        long_term.append(superbloco)
        salvar_json(LONG_TERM_PATH, long_term)

        # Remove os 20 blocos mais antigos
        blocos = blocos[20:]
        salvar_json(COMPILED_BLOCKS_PATH, blocos)
        
        # Limita superblocos a no máximo 50
        if len(long_term) > 50:
            long_term = long_term[-50:]
            salvar_json(LONG_TERM_PATH, long_term)

    return True if len(blocos) >= 36 else False

