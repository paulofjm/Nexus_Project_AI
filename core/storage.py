import os
import json
import faiss
from langchain_community.vectorstores import FAISS
import numpy as np
from datetime import datetime
# from utils.logger import write_monitor_log as log
import pygetwindow as gw
import os
import re
import win32process
import psutil

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

def detect_active_project_root(fallback=BASE_DIR):
    """
    Detecta a raiz do projeto com uma abordagem híbrida e definitiva.
    1. Coleta todos os diretórios de trabalho (CWDs) válidos de processos de IDE.
    2. Usa o título da janela ativa para selecionar o CWD correto da lista.
    """
    if os.name != 'nt' or not gw or not win32process or not psutil:
        print("Aviso: Bibliotecas necessárias (pygetwindow, psutil, pywin32) não encontradas. Usando fallback.")
        return fallback

    try:
        ide_exe_names = {'code.exe', 'pycharm64.exe'}
        invalid_path_segments = {'system32', 'appdata', 'program files', 'windows', '.vscode', 'extensions'}
        
        # Etapa 1 e 2: Coleta todos os CWDs de projeto válidos de todos os processos de IDE.
        valid_project_paths = set()
        for proc in psutil.process_iter(['name', 'cwd']):
            try:
                if proc.info['name'].lower() in ide_exe_names:
                    path = proc.info['cwd']
                    if not path: continue
                    
                    path_lower = path.lower()
                    if not any(invalid_segment in path_lower for invalid_segment in invalid_path_segments):
                        valid_project_paths.add(path)
            except (psutil.Error, AttributeError):
                continue
        
        if not valid_project_paths:
            print("Aviso: Nenhum processo de IDE com um CWD de projeto válido foi encontrado.")
            return fallback

        # Etapa 3: Usa a janela ativa para obter o contexto do usuário.
        active_window = gw.getActiveWindow()
        # Se não houver janela ativa ou a lista de projetos válidos tiver apenas um item, retorne a escolha mais óbvia.
        if not active_window or not active_window.title or len(valid_project_paths) == 1:
            # Retorna o único encontrado ou o mais curto (mais provável de ser a raiz).
            best_guess = min(valid_project_paths, key=len)
            print(f"Debug: Usando melhor palpite (sem correspondência de janela): {best_guess}")
            return best_guess

        active_title_lower = active_window.title.lower()
        
        # Etapa 4: Cruza o título da janela com os caminhos de projeto encontrados.
        best_match = None
        highest_match_score = 0

        for path in valid_project_paths:
            # Pega o nome da pasta final do caminho, ex: "Nexus_project"
            project_folder_name = os.path.basename(path).lower()
            
            # Se o nome da pasta do projeto estiver no título da janela ativa...
            if project_folder_name in active_title_lower:
                # ...damos preferência à correspondência mais longa, para evitar ambiguidades.
                if len(project_folder_name) > highest_match_score:
                    highest_match_score = len(project_folder_name)
                    best_match = path

        if best_match:
            print(f"Debug: Projeto detectado por correspondência de título '{active_window.title[:60]}...' -> {best_match}")
            return best_match
        
        # Se nenhuma correspondência com o título for encontrada, retorna o melhor palpite.
        best_guess = min(valid_project_paths, key=len)
        print(f"Aviso: Título da janela ativa não correspondeu a nenhum projeto. Usando melhor palpite: {best_guess}")
        return best_guess

    except Exception as e:
        print(f"Ocorreu um erro inesperado na detecção: {e}. Usando fallback.")
        return fallback


# Diretórios

LOG_DIR = os.path.join(BASE_DIR, "..", "data", "logs")
TEMP_DIR = os.path.join(BASE_DIR, "..", "data", "temp")
FAISS_DIR = os.path.join(LOG_DIR, "faiss")
SCREENSHOT_DIR = os.path.join(LOG_DIR, "screenshots")
# code_summary_path = os.path.abspath(os.path.join("data", "logs", "project_code_summary.txt"))
MONITOR_LOG_PATH = os.path.abspath(os.path.join(LOG_DIR, "monitor_log.json"))
CHAT_HISTORY_PATH = os.path.abspath(os.path.join(LOG_DIR, "chat_history.json"))
CHAT_TITLES_FILE = os.path.abspath(os.path.join(LOG_DIR, "chat_titles.json"))
TURNS_PATH = os.path.abspath(os.path.join(LOG_DIR, "turns.json"))
CODE_VERSION = os.path.abspath(os.path.join(LOG_DIR, "code_versions"))
# Arquivos de metadados e índice FAISS
COMPILED_BLOCKS_PATH = os.path.join(LOG_DIR, "activity_compiled_blocks.json")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "dev_activity_index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "dev_activity_metadata.json")
CHAT_HISTORY_FILE = os.path.join(LOG_DIR, "chat_history.json")
ASSETS_DIR = os.path.join(BASE_DIR, "..", "gui", "assets")

# Criação de diretórios (ordem: pais → filhos)
os.makedirs(os.path.abspath(LOG_DIR), exist_ok=True)
os.makedirs(os.path.abspath(TEMP_DIR), exist_ok=True)
os.makedirs(os.path.abspath(FAISS_DIR), exist_ok=True)
os.makedirs(os.path.abspath(SCREENSHOT_DIR), exist_ok=True)
os.makedirs(os.path.abspath(CODE_VERSION), exist_ok=True)

# Carregamento do FAISS e Metadata
def load_faiss_and_metadata():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # print("FAISS e Metadata carregados.")
    else:
        dimension = 384  # Tamanho do vetor de embedding
        index = faiss.IndexFlatL2(dimension)
        metadata = []
        # print("Criando novo FAISS e Metadata.")
    return index, metadata

faiss_index, metadata = load_faiss_and_metadata()

# Salvar FAISS e metadata
def save_faiss_and_metadata():
    from utils.logger import write_monitor_log as log
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log("FAISS e Metadata atualizados.")

# Novo método: adicionar embedding para um bloco compilado
def adicionar_bloco_ao_faiss(embedding_vector, bloco_compilado):
    """
    Adiciona um vetor de embedding (np.array) ao índice FAISS
    e atualiza o metadata.
    """
    if "nivel" not in bloco_compilado:
        bloco_compilado["nivel"] = "curto" 
    faiss_index.add(np.array([embedding_vector], dtype=np.float32))
    metadata.append(bloco_compilado)
    save_faiss_and_metadata()


