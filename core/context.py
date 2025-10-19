import os
import sys
# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import re
import psutil
from datetime import datetime
from core.storage import metadata, faiss_index
from models.embedding import embed_text
import hashlib
import glob
import faiss
from core.storage import FAISS_INDEX_PATH, METADATA_PATH, CODE_VERSION
import json
from utils.logger import write_monitor_log as log
from models.llm_manager import ask_with_model
from models.llm_config import NODE_MODELS_DEFAULT

# --- Refina query de forma semântica ---
def refinar_query_semanticamente(query: str, turnos: list[dict]) -> str:
    contexto = "\n".join(
        f"Usuário: {t['pergunta']}\nAgente: {t['resposta']}"
        for t in turnos[-5:]
    )
    prompt = f"""
    Dado o histórico e a nova pergunta, reescreva a consulta para busca vetorial.
    Histórico: {contexto}
    Nova pergunta: {query}
    Retorne apenas a consulta refinada.
    """.strip()
    # 1. Passa nome e formato separadamente.
    model_name = NODE_MODELS_DEFAULT["router"]["name"]
    model_format = NODE_MODELS_DEFAULT["router"]["format"]
    
    response_data = ask_with_model(
        model_name,
        model_format,
        user_prompt=prompt,
        system_prompt="Você é um assistente que reescreve perguntas para busca semântica."
    )
    
    # 2. Extrai o conteúdo do dicionário de resposta.
    resposta_refinada = response_data.get("content", query)
    return resposta_refinada.strip()


# --- Busca no FAISS e retorna SÓ os índices dos blocos mais relevantes ---
def consultar_faiss(query: str, k: int = 5, preferencia: str = "curto>medio>longo", filtros: dict = None) -> list[int]:
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return []

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Filtro estruturado, se solicitado (opcional, pode ser removido se não usar)
    if filtros:
        def match(item):
            for chave, valor in filtros.items():
                if valor not in str(item.get(chave, "")):
                    return False
            return True
        filtered = [i for i, b in enumerate(metadata) if match(b)]
    else:
        filtered = list(range(len(metadata)))

    if not filtered:
        return []

    # Busca vetorial normal nos blocos filtrados
    query_vec = embed_text([query])
    query_vec = np.array(query_vec).astype("float32")
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    # Importante: se usou filtro, precisamos buscar nos índices originais!
    # Por padrão busca nos 50 primeiros e depois "mapeia" para os índices corretos
    dists, indices = index.search(query_vec, 50)
    ranked = []
    for dist, i in zip(dists[0], indices[0]):
        if isinstance(i, int) and i < len(metadata):
            if not filtros or i in filtered:
                ranked.append((i, dist))
    # Ordenação por preferência hierárquica e score
    ordem = preferencia.split(">")
    ranked.sort(key=lambda x: (ordem.index(metadata[x[0]].get("nivel", "longo")), x[1]))

    # Retorna só os índices dos top-k blocos relevantes
    return [i for i, _ in ranked[:k]]

# --- Recupera blocos a partir de lista de índices ---
def get_blocks_by_indices(indices: list[int]):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return [metadata[i] for i in indices if isinstance(i, int) and i < len(metadata)]




# Função para obter um resumo dos arquivos do projeto
def get_project_files_summary(project_root, max_files=10):
    summaries = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(('.py', '.js', '.html', '.css', '.json', '.md')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(5000) # Limita a leitura a n caracteres
                        summaries.append(f"\n📄 {file}:\n{content}")
                        if len(summaries) >= max_files:
                            return "\n".join(summaries)
                except Exception:
                    continue
    return "\n".join(summaries)

# Diretórios que serão sempre ignorados (podem ser expandidos)
SKIP_DIRS = {'local', '.git', '.venv', 'venv', '__pycache__', 'node_modules', '.idea', '.vscode', 'build', 'dist', '.mypy_cache', '.pytest_cache'}
# Extensões de código permitidas (ajuste conforme seu projeto)
CODE_EXTS = ('.py', '.ipynb', '.js', '.ts', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.rs', '.php')


# Função para armazenar um compilado do código do projeto em um arquivo
def generate_compiled_code(
    project_root: str | None = None,
    allowed_dirs: set[str] | None = None,
    allowed_files: set[str] | None = None,
    exts: tuple[str] = CODE_EXTS,
    # max_file_size_bytes: int = 1_000_000  # 1MB
) -> str:
    """
    Percorre o projeto e gera um snapshot concatenado de código, ignorando diretórios irrelevantes.
    """
    if project_root is None:
        from core.storage import detect_active_project_root
        log("Aviso: 'project_root' não foi fornecido. Tentando detectar o projeto da janela ativa...")
        project_root = detect_active_project_root()
        if project_root:
            log(f"Projeto detectado: {project_root}")
        else:
            log("Erro: Não foi possível detectar o projeto. Forneça o 'project_root' explicitamente.")
            return "" # Retorna vazio ou lança um erro
    allowed_dirs = set(allowed_dirs or [])
    allowed_files = set(allowed_files or [])

    parts = []

    for root, dirs, files in os.walk(project_root):
        # Remove dirs a serem ignorados
        # dirs[:] = [d for d in dirs if all(skip not in os.path.join(root, d) for skip in SKIP_DIRS) and not d.startswith('.')]
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        rel_path = os.path.relpath(root, project_root)

        # Filtro por allowed_dirs
        if allowed_dirs and rel_path.split(os.sep)[0] not in allowed_dirs:
            continue

        for file in files:
            if allowed_files and file not in allowed_files:
                continue
            if not file.endswith(exts):
                continue
            file_path = os.path.join(root, file)
            # if os.path.getsize(file_path) > max_file_size_bytes:
            #     continue  # Ignora arquivos grandes demais

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                parts.append(f"# ==== {os.path.relpath(file_path, project_root)} ====\n{content}\n")
            except Exception:
                continue


    # Concatena e salva o snapshot
    full_code = "\n".join(parts)
    hash_digest = hashlib.sha256(full_code.encode('utf-8')).hexdigest()[:10]
    BASE_LOGS = os.path.join(os.path.dirname(__file__), "..", "data", "logs", "code_versions")
    project_slug = os.path.basename(os.path.abspath(project_root))
    version_dir = os.path.join(BASE_LOGS, project_slug)
    os.makedirs(version_dir, exist_ok=True)
    out_pattern = os.path.join(version_dir, f"code_summary_*_{hash_digest}.txt")
    # Se já existe snapshot com esse hash, não cria novo
    if glob.glob(out_pattern):
        # Retorna o path do existente
        return glob.glob(out_pattern)[0]

    # Se não existe, cria um novo com timestamp + hash
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(version_dir, f"code_summary_{stamp}_{hash_digest}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(full_code)
    return out_path

def clean_code_for_context(code: str) -> str:
    import re
    code = re.sub(r'(""".*?""")|(\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)

    # Prefixos para ignorar
    ignore_prefixes = ['#', 'import ', 'from ', 'print(', 'def test_', 'class Test']

    # Palavras/chunks: se aparecer na linha, troca tudo por 'Removido para contexto'
    remove_keywords = ['log(', 'warnings', 'threading', 'time', 'sys', 'core.window_tracker', 'track_activity',
                       'utils', 'code_monitor', 'nodes_graph.langgraph_nodes', 'graph_executor', 'models.llm_manager',
                       'load_llm', 'models.llm_config', 'contexto(']

    lines = code.splitlines()
    filtered = []
    for line in lines:
        l = line.strip()
        # Ignora linha vazia ou prefixos
        if not l or any(l.startswith(prefix) for prefix in ignore_prefixes):
            continue
        # Remove comentários inline
        l_no_comment = l.split('#')[0].rstrip()
        if not l_no_comment:
            continue
        # Se linha contém alguma keyword do remove_keywords, substitui por "Removido para contexto"
        if any(kw in l_no_comment for kw in remove_keywords):
            filtered.append("Removido para contexto")
            continue
        filtered.append(l_no_comment)
    return '\n'.join(filtered)

def load_latest_code_summary(project_slug: str | None = None) -> str:
    """
    Return the content of the most recent code summary file.
    Se project_slug não for passado, busca em todos os subdiretórios.
    """
    # Garante que não existe espaço acidental no slug
    if project_slug:
        project_slug = project_slug.strip()
        pattern = os.path.join(CODE_VERSION, project_slug, "*.txt")
    else:
        pattern = os.path.join(CODE_VERSION, "*", "*.txt")

    candidates = glob.glob(pattern)
    if not candidates:
        return "não encontrado"

    # Ordena pelo arquivo mais recente
    newest = max(candidates, key=os.path.getmtime)
    try:
        with open(newest, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return log(f"erro ao abrir o arquivo: {e}")
    
# Função para resumir arquivo compilado de código para contexto.
def resumir_arquivo_compilado():
    """Carrega o snapshot de código mais recente e pede a um LLM para resumi-lo."""
    codigo_completo = load_latest_code_summary()
    if not codigo_completo or codigo_completo == "não encontrado":
        return "Nenhum arquivo de código para resumir foi encontrado."

    MAX_WORDS = 3000
    system_prompt = f"Faça um resumo de até {MAX_WORDS} palavras do código do projeto fornecido para que o contexto do projeto possa ser entendido. Não inclua exemplos de código, apenas uma descrição do que o código faz. O resumo NÃO deve passar de {MAX_WORDS} palavras."
    user_content = f"<codigo_do_projeto>{codigo_completo}</codigo_do_projeto>"

    try:
        # 1. Passa nome e formato do modelo de CÓDIGO separadamente.
        model_name = NODE_MODELS_DEFAULT["code"]["name"]
        model_format = NODE_MODELS_DEFAULT["code"]["format"]

        response_data = ask_with_model(
            model_name,
            model_format,
            user_prompt=user_content,
            system_prompt=system_prompt
        )
        
        # 2. Extrai o conteúdo do dicionário de resposta.
        resumo = response_data.get("content", f"[Erro ao resumir o código]")

        # Validação do tamanho (opcional, mas bom manter)
        if len(resumo.split()) > MAX_WORDS:
            resumo = ' '.join(resumo.split()[:MAX_WORDS])

    except Exception as e:
        resumo = f"[Erro excepcional ao resumir o código: {e}]"
    
    return resumo

# Função para detectar processos abertos
def check_running_processes():
    process_list = []
    for proc in psutil.process_iter(['name']):
        try:
            process_list.append(proc.info['name'].lower())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return process_list

# Função para recuperar contexto com base na consulta
def retrieve_context(query, top_k=5):
    query_embedding = embed_text([query])
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)

    running_processes = check_running_processes()

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result['distance'] = float(dist)

            # Análise de OCR + processos
            extracted_text = result.get('extracted_text', '').lower()
            notes = []

            if any(keyword in extracted_text for keyword in ["vscode", "visual studio code", "py", "jupyter", "anaconda", "notebook", "explorer", "terminal", "python"]):
                notes.append("Ambiente de desenvolvimento detectado no OCR.")

            if any(proc in running_processes for proc in ["code.exe", "python.exe", "jupyter-notebook.exe"]):
                notes.append("Processo ativo de desenvolvimento identificado.")

            if notes:
                result['detection_notes'] = " ".join(notes)

            results.append(result)

    return results

# Função para formatar os resultados do contexto
def format_context(context_results):
    formatted = ""
    for i, item in enumerate(context_results, 1):
        formatted += f"{i}.\n"

        # Exibir o timestamp de forma formatada, se disponível
        if item.get('timestamp'):
            try:
                ts = datetime.fromisoformat(item['timestamp'])
                formatted += f"Horário: {ts.strftime('%d/%m/%Y %H:%M:%S')}\n"
            except Exception:
                formatted += f"Horário (bruto): {item['timestamp']}\n"

        formatted += f"🪟 Janela: {item.get('active_window', 'Não disponível')}\n"

        if item.get('typed_text'):
            formatted += f"⌨Texto Digitado: {item['typed_text']}\n"
        if item.get('extracted_text'):
            formatted += f"OCR (trecho): {item['extracted_text'][:500]}...\n"
        if item.get('detection_notes'):
            formatted += f"Notas: {item['detection_notes']}\n"

        formatted += "-" * 50 + "\n"

    return formatted.strip()

# Função para montar o prompt inteligente
def build_prompt(user_query, context_blks):
    formatted = "\n".join(
        f"{i+1}. [{blk.get('timestamp', '')}] {blk.get('context_summary','')[:200]} (Janela: {blk.get('active_window','')})"
        for i, blk in enumerate(context_blks)
    )
    final_prompt = (
        "Você é um assistente focado em analisar atividades de desenvolvimento.\n"
        "Use apenas o contexto relevante abaixo para responder de forma precisa.\n"
        "Se não houver informação suficiente, informe explicitamente.\n\n"
        f"Contexto relevante:\n{formatted}\n\n"
        f"Pergunta: {user_query}\nResposta:"
    )
    return final_prompt

