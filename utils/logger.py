import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from datetime import datetime
from core.storage import MONITOR_LOG_PATH, CHAT_HISTORY_PATH, CHAT_HISTORY_PATH


def write_monitor_log(message: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "message": message
    }

    os.makedirs(os.path.dirname(MONITOR_LOG_PATH), exist_ok=True)

    try:
        if os.path.exists(MONITOR_LOG_PATH):
            with open(MONITOR_LOG_PATH, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(MONITOR_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, indent=2, ensure_ascii=False)
    except Exception:
        pass  # silencia erros de log


def salvar_resposta_chat(pergunta: str, resposta: str, modelo: str):
    registro = {
        "timestamp": datetime.now().isoformat(),
        "modelo_utilizado": modelo,
        "pergunta": pergunta,
        "resposta": resposta
    }

    os.makedirs(os.path.dirname(CHAT_HISTORY_PATH), exist_ok=True)

    try:
        if os.path.exists(CHAT_HISTORY_PATH):
            with open(CHAT_HISTORY_PATH, "r+", encoding="utf-8") as f:
                historico = json.load(f)
                historico.append(registro)
                f.seek(0)
                json.dump(historico, f, indent=2, ensure_ascii=False)
        else:
            with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump([registro], f, indent=2, ensure_ascii=False)
    except Exception:
        pass
