import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from core.storage import CHAT_HISTORY_FILE

# Salvar histórico de chats
def save_chat_history(chats):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Falha ao salvar chats: {e}")
        
# Carregar histórico de chats
def load_chat_history():
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                chats = json.load(f)
        else:
            chats = {"Chat 1": []}
    except Exception as e:
        print(f"[WARN] Falha ao carregar chats: {e}")
        chats = {"Chat 1": []}
    return chats


