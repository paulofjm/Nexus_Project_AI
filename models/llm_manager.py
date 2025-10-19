import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from utils.logger import write_monitor_log as log
# from langchain_community.chat_models import ChatOllama # Importa o ChatOllama
from .llm_base import (GGUFLLM, TransformersLLM, OpenAILLM)
from .llm_config import (
    LLM_LOCAL_CACHE,
    OPENAI_API_KEY,
)
import openai
_active_llm = {}

# Cache para manter os modelos carregados em memória
_active_llm_cache = {}

# ==== CARREGADOR UNIFICADO ====
def load_llm(model_name: str, model_format: str):
    """Carrega e armazena em cache uma instância de modelo LLM."""
    cache_key = f"{model_name}_{model_format}"
    if cache_key in _active_llm_cache:
        return _active_llm_cache[cache_key]

    log(f"LLM_MANAGER: Carregando modelo '{model_name}' (Formato: {model_format})...")
    
    if model_format == "openai":
        # Passa o 'model_name' para o construtor da OpenAILLM
        model = OpenAILLM(model_name=model_name)
    
    elif model_format == "ollama":
        model = ChatOllama(model=model_name, temperature=0.1)
    
    else:
        raise ValueError(f"Formato de modelo desconhecido: '{model_format}'")

    _active_llm_cache[cache_key] = model
    log(f"LLM_MANAGER: Modelo '{model_name}' carregado com sucesso.")
    return model

# def ask_with_model(model_name: str, model_format: str, user_prompt, system_prompt: str, is_json_output: bool = False):
#     """
#     Função universal e robusta para consultar qualquer modelo.
#     """
#     try:
#         # Carrega o modelo correto, aplicando o modo JSON se necessário
#         if model_format == "ollama" and is_json_output:
#             # Cria uma instância específica para garantir o modo JSON
#             model = ChatOllama(model=model_name, temperature=0.1, format="json")
#         else:
#             model = load_llm(model_name, model_format)

#         # Constrói a lista de mensagens de forma segura e padronizada
#         messages = [{"role": "system", "content": system_prompt}]
#         if isinstance(user_prompt, list):
#             messages.extend(user_prompt)
#         elif isinstance(user_prompt, str):
#             messages.append({"role": "user", "content": user_prompt})
#         else:
#             raise TypeError(f"Tipo de user_prompt inválido: {type(user_prompt)}")

#         log(f"LLM_MANAGER: Consultando modelo -> {model_name} (Formato: {model_format})")
#         start_time = time.time()
        
#         response = model.invoke(messages)
        
#         end_time = time.time()
#         duration = end_time - start_time
        
#         content = response.content if hasattr(response, "content") else str(response)
        
#         log(f"LLM_MANAGER: Resposta recebida em {duration:.2f} segundos.")
        
#         return {"content": content, "duration": duration}

#     except Exception as e:
#         log(f"LLM_MANAGER: ERRO ao consultar o modelo ({model_name}): {e}")
#         return {"content": f"Ocorreu um erro: {e}", "duration": 0}

def ask_with_model(model_name: str, model_format: str, user_prompt, system_prompt: str, is_json_output: bool = False):
    """
    Função universal e robusta para consultar qualquer modelo.
    """
    try:
        # Se for um modelo Ollama E a saída esperada for JSON,
        # ele cria uma instância específica com o modo JSON ativado.
        if model_format == "ollama" and is_json_output:
            model = ChatOllama(model=model_name, temperature=0.1, format="json")
        else:
            # Para todos os outros casos, usa o carregador padrão com cache.
            model = load_llm(model_name, model_format)

        # Constrói a lista de mensagens de forma segura
        messages = [{"role": "system", "content": system_prompt}]
        if isinstance(user_prompt, list):
            messages.extend(user_prompt)
        elif isinstance(user_prompt, str):
            messages.append({"role": "user", "content": user_prompt})
        else:
            raise TypeError(f"Tipo de user_prompt inválido: {type(user_prompt)}")

        log(f"LLM_MANAGER: Consultando modelo -> {model_name} (Formato: {model_format}, JSON: {is_json_output})")
        start_time = time.time()
        
        response = model.invoke(messages)
        
        end_time = time.time()
        duration = end_time - start_time
        
        content = response.content if hasattr(response, "content") else str(response)
        
        log(f"LLM_MANAGER: Resposta recebida em {duration:.2f} segundos.")
        
        return {"content": content, "duration": duration}

    except Exception as e:
        log(f"LLM_MANAGER: ERRO ao consultar o modelo ({model_name}): {e}")
        return {"content": f"Ocorreu um erro: {e}", "duration": 0}    
    
# ==== CONSULTA PADRÃO ====

def ask_llm(llm_type: str, prompt: str, system_prompt: str = "Você é um analista de atividades, considerando tanto o histórico de perguntas quanto o contexto FAISS.") -> str:
    if not _active_llm:
        raise RuntimeError("LLM ainda não carregado. Use `load_llm()` primeiro.")

    try:
        if llm_type  == "openai":
            # OpenAI aceita mensagens formatadas
            response = _active_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
            return response.content if hasattr(response, "content") else response

        elif llm_type  in ["transformers", "gguf"]:
            # Modelos locais só aceitam texto direto
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = _active_llm.invoke(full_prompt)
            return response.content if hasattr(response, "content") else response

        else:
            return "Modo de modelo não reconhecido."

    except Exception as e:
        return f"Erro ao consultar o modelo: {e}"



def check_api_key(api_key: str) -> bool:
    """Verifica se a chave de API da OpenAI é válida."""
    if not api_key: return False
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        log("INFO: Chave de API da OpenAI validada com sucesso.")
        return True
    except openai.AuthenticationError:
        log("ERRO: Falha na autenticação da OpenAI. Chave de API inválida.")
        return False
    except Exception as e:
        log(f"ERRO: Ocorreu um erro inesperado ao verificar a chave da OpenAI: {e}")
        return False