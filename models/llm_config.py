import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv 
# from utils.logger import write_monitor_log
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_LOCAL_CACHE = os.getenv("LLM_LOCAL_CACHE", "").strip()

# Carrega configurações do .env
# LLM_MODE = os.getenv("LLM_MODE", "").strip().lower()
# LLM_LOCAL_ID = os.getenv("LLM_LOCAL_ID", "").strip()
# LLM_LOCAL_FILENAME = os.getenv("LLM_LOCAL_FILENAME", "").strip()
# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL", "").strip()
# MODEL_CODE = os.getenv("MODEL_CODE", "deepseek-coder:6.7b-instruct").strip().lower()
# MODEL_CODE_FORMAT = os.getenv("MODEL_CODE_FORMAT", "ollama").strip().lower()
# MODEL_ACTIVITY = os.getenv("MODEL_ACTIVITY", "gemma:2b").strip().lower()
# MODEL_ACTIVITY_FORMAT = os.getenv("MODEL_ACTIVITY_FORMAT", "ollama").strip().lower()
# MODEL_WINDOW = os.getenv("MODEL_WINDOW", "").strip().lower()
# MODEL_WINDOW_FORMAT = os.getenv("MODEL_WINDOW_FORMAT", "").strip().lower()
# MODEL_PRINCIPAL = os.getenv("GLOBAL_AGENT", "gpt-4o-mini").strip().lower()
# MODEL_PRINCIPAL_FORMAT = os.getenv("GLOBAL_AGENT_FORMAT", "openai").strip().lower()
# MODEL_ROUTER = os.getenv("MODEL_ROUTER", "gemma:2b").strip().lower()
# MODEL_ROUTER_FORMAT = os.getenv("MODEL_ROUTER_FORMAT", "ollama").strip().lower()
# MODEL_SUMMARIZER_CODE = os.getenv("MODEL_SUMMARIZER_CODE", "").strip().lower()
# MODEL_SUMMARIZER_CODE_FORMAT = os.getenv("MODEL_SUMMARIZER_CODE_FORMAT", "").strip().lower()


# --- Modelos Padrão para Cada Nó ---

MODEL_ROUTER_DEFAULT = os.getenv("MODEL_ROUTER", "gemma:2b").strip()
MODEL_ACTIVITY_DEFAULT = os.getenv("MODEL_ACTIVITY", "gemma:2b").strip()
MODEL_CODE_DEFAULT = os.getenv("MODEL_CODE", "deepseek-coder:6.7b-instruct").strip()
MODEL_PRINCIPAL_DEFAULT = os.getenv("MODEL_PRINCIPAL", "gpt-4o-mini").strip()

# --- Formatos Padrão ---
# Ajuda o sistema a saber como formatar o prompt para cada modelo.
def get_format_from_name(model_name: str) -> str:
    """Determina o formato ('openai' ou 'ollama') com base no nome do modelo."""
    if "gpt-" in model_name:
        return "openai"
    return "ollama"

# --- Dicionário Central de Configurações Padrão ---
NODE_MODELS_DEFAULT = {
    "router": {"name": MODEL_ROUTER_DEFAULT, "format": get_format_from_name(MODEL_ROUTER_DEFAULT)},
    "activity": {"name": MODEL_ACTIVITY_DEFAULT, "format": get_format_from_name(MODEL_ACTIVITY_DEFAULT)},
    "code": {"name": MODEL_CODE_DEFAULT, "format": get_format_from_name(MODEL_CODE_DEFAULT)},
    "principal": {"name": MODEL_PRINCIPAL_DEFAULT, "format": get_format_from_name(MODEL_PRINCIPAL_DEFAULT)},
}

# --- Modelos Disponíveis para a Interface Gráfica ---
# "Nome Amigável na GUI": "ID Real do Modelo"
AVAILABLE_MODELS = {
    "GPT-4o-mini (OpenAI)": "gpt-4o-mini",
    "GPT-4o (OpenAI)": "gpt-4o",
    "Gemma 2B (Ollama Local)": "gemma:2b",
    "DeepSeek Coder 6.7B (Ollama Local)": "deepseek-coder:6.7b-instruct",
    "Llama 3.1 8B (Ollama Local)": "llama3.1:8b",
    "Phi-3 Mini (Ollama Local)": "phi3:mini"
}

# def set_selected_model(model_name: str, api_key: str | None = None):
#     """
#     Atualiza as variáveis globais com o modelo e a chave de API selecionados pelo usuário.
#     """
#     global SELECTED_MODEL, API_KEY
    
#     if model_name in AVAILABLE_MODELS:
#         SELECTED_MODEL = AVAILABLE_MODELS[model_name]
#         API_KEY = api_key
#         print(f"INFO: Modelo alterado para: {SELECTED_MODEL}")
#     else:
#         print(f"ERRO: Modelo '{model_name}' não encontrado em AVAILABLE_MODELS.")

