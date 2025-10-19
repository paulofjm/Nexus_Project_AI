from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import time
import json
import os
from core.context import consultar_faiss, load_latest_code_summary, get_blocks_by_indices
from models.llm_manager import ask_with_model, load_llm
from models.llm_config import (
    NODE_MODELS_DEFAULT 
)

from utils.logger import write_monitor_log as log
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore as KeyValueStore
from .nexus_tools_retriever import TOOLS_NEXUS
from core.storage import METADATA_PATH, TURNS_PATH
from datetime import date, timedelta



# Define a pasta raiz do projeto de forma consistente com a GUI
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ========== PERSISTÊNCIA DA MEMÓRIA MULTI-TURN ==========

def get_turnos(store: KeyValueStore, session_id: str): # -> List[dict]:
    retrieved_value = store.mget([session_id])
    return retrieved_value[0] if retrieved_value and retrieved_value[0] is not None else []

def set_turnos(store: KeyValueStore, session_id: str, turnos: List[dict]):
    store.mset([(session_id, turnos)])

def save_turnos_to_disk(store: KeyValueStore, session_ids: List[str] = ["default"]):
    os.makedirs(os.path.dirname(TURNS_PATH), exist_ok=True)
    all_keys = list(store.yield_keys())
    all_values = store.mget(all_keys)
    all_turnos = dict(zip(all_keys, all_values))
    with open(TURNS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_turnos, f, ensure_ascii=False, indent=2)

# Sugestão para load_turnos_from_disk
def load_turnos_from_disk(store: KeyValueStore):
    if not os.path.exists(TURNS_PATH):
        return
    try:
        with open(TURNS_PATH, 'r', encoding='utf-8') as f:
            all_turnos = json.load(f)
        items_to_set = list(all_turnos.items())
        if items_to_set:
            store.mset(items_to_set)
    except (json.JSONDecodeError, IOError) as e:
        log(f"AVISO: Não foi possível carregar o histórico de turnos de '{TURNS_PATH}'. Arquivo pode estar corrompido ou inacessível. Erro: {e}")
        # Opcional: fazer backup do arquivo corrompido
        # os.rename(TURNS_PATH, f"{TURNS_PATH}.bak")

# ========== ESTADO COMPARTILHADO ==========

store = KeyValueStore()
load_turnos_from_disk(store)

class NexusState(TypedDict, total=False):
    pergunta: str
    tipo: Optional[str]
    topk_indices: Optional[List[int]]
    resposta: Optional[str]
    session_id: Optional[str]
    inicio: Optional[str]
    fim: Optional[str]
    detalhes: Optional[str]
    tentativas: Optional[int]
    
    # Campos para carregar a configuração de modelo dinâmica
    model_router_name: Optional[str]
    model_router_format: Optional[str]
    model_activity_name: Optional[str]
    model_activity_format: Optional[str]
    model_code_name: Optional[str]
    model_code_format: Optional[str]
    model_principal_name: Optional[str]
    model_principal_format: Optional[str]

    last_processing_time: Optional[float]
    total_processing_time: Optional[float]

historico_fluxo = []

# ========== AGENTE DE FERRAMENTAS ==========

llm_tools = ChatOpenAI(model=NODE_MODELS_DEFAULT["principal"]["name"], temperature=0)
tool_graph_executor = create_react_agent(llm_tools, TOOLS_NEXUS)

# ========== NODES DO GRAFO ==========

def node_processar_query(state: NexusState) -> NexusState:
    pergunta = state["pergunta"].strip()
    session_id = state.get("session_id", "default")
    inicio = time.time()
    log(f"Node: processar_query | Pergunta recebida: '{pergunta}'")
    historico_fluxo.append({"nodo": "processar_query", "inicio": inicio, "pergunta": pergunta, "fim": time.time()})
    return {**state, "pergunta": pergunta, "total_processing_time": 0.0, "session_id": session_id}

# Em langgraph_nodes.py

def node_interpretador(state: NexusState) -> NexusState:
    pergunta = state["pergunta"]
    hoje = date.today()
    # ... (o 'agent_instructions' e 'user_message_list' continuam iguais)
    agent_instructions = f"""
    Data de referência: {date.today().isoformat()}. 
    <instrução indispensavel>
    Responda APENAS com um JSON válido em formato ISO8601.. 
    Classifique a intenção, USE APENAS UMA DAS OPÇÕES a seguir: 'mais_recente', 'temporal', 'semantica', 'codigo', ou 'fallback'. 
    Se 'temporal', extraia 'inicio' e 'fim'.
    
    <instruçoes gerais>
    - Responda unica e exclusivamente com um JSON válido, sem explicações.
    - Não use blocos de código, apenas o JSON puro, sem ```json.
    - Classifique a intenção da pergunta abaixo. Responda usando este formato JSON.
    - Para perguntas sobre a atividade mais recente, como "o que eu fiz por último?" ou "qual foi minha última atividade?", use "mais_recente".
    - Para perguntas sobre atividades em um período de tempo específico (ex: "últimas 3 horas", "ontem"), use "temporal" e extraia o intervalo de tempo em formato ISO8601.
    - Para outras perguntas que necessitem de busca por similaridade, use "semantica".
    - Para perguntas sobre código (sugestões, bugs, explicações, transcrições do projeto), use "codigo".
    - Para comandos que executam ferramentas (abrir, listar, printar), use "tools".
    - Para saudações e outros casos, use "fallback".
    - Use apenas um "tipo" por vez, sem misturar.
    - Jamais use "fallback" se puder usar outro tipo.
    - Responda unica e exclusivamente com um JSON válido, sem explicações adicionais.
    - Use a data de referência para calcular datas relativas como "hoje", "ontem", etc.
    Exemplos com base na data de referência:
    - Pergunta "o que fiz ontem?" -> inicio: "{(hoje - timedelta(days=1)).isoformat()}T00:00:00", fim: "{(hoje - timedelta(days=1)).isoformat()}T23:59:59"
    - Pergunta "minhas atividades de hoje" -> inicio: "{hoje.isoformat()}T00:00:00", fim: "{hoje.isoformat()}T23:59:59"

    - "tipo": ""
    - "inicio": "YYYY-MM-DDTHH:MM:SS" (obrigatório se for temporal)
    - "fim": "YYYY-MM-DDTHH:MM:SS" (obrigatório se for temporal)
    - "detalhes": "outros detalhes úteis"
    """
     
    model_name = state.get("model_router_name", NODE_MODELS_DEFAULT["router"]["name"])
    model_format = state.get("model_router_format", NODE_MODELS_DEFAULT["router"]["format"])

    user_content = f"Pergunta do usuário: {pergunta}"
    user_message_list = [{"role": "user", "content": user_content}]

    log(f"Node: interpretador | Usando modelo: {model_name} ({model_format})")

    response_data = ask_with_model(model_name, 
                                   model_format, 
                                   user_prompt=user_message_list,
                                   system_prompt=agent_instructions,
                                   is_json_output=True)
    


    content = response_data.get("content", {})
    duration = response_data.get("duration", 0)
    current_total_time = state.get("total_processing_time", 0.0)
    new_total_time = current_total_time + duration
    parsed = json.loads(content) if isinstance(content, str) else content if isinstance(content, dict) else {}
    if 'tipo' not in parsed: parsed = {"tipo": "fallback", "detalhes": "Falha ao interpretar a intenção."}
        
    log(f"Node: interpretador | Intent: {parsed.get('tipo')} | Duração: {new_total_time:.2f}s")
    return {**state, **parsed, "total_processing_time": new_total_time, "last_processing_time": duration}

# ==== CONSULTAR MEMÓRIA ====
def node_consultar_memoria(state: NexusState) -> NexusState:
    tipo = state.get("tipo", "semantica")
    pergunta = state["pergunta"]
    topk_indices = []

    # Consulta temporal ou "mais recente"
    if tipo == "temporal" and state.get("inicio") and state.get("fim"):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        inicio = state["inicio"]
        fim = state["fim"]
        # Pegue índices dos blocos que caem no intervalo
        indices_relevantes = [
            i for i, b in enumerate(meta)
            if inicio <= b.get("timestamp", "") <= fim
        ]
        topk_indices = indices_relevantes[:8]  # Limite hard, se quiser parametrizar fique à vontade
        log(f"Node: consultar_memoria | Temporal: {len(topk_indices)} blocos no período {inicio} até {fim}.")
    elif tipo == "mais_recente":
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta:
            topk_indices = [len(meta) - 1]  # Só o último bloco
        else:
            topk_indices = []
        log(f"Node: consultar_memoria | Mais recente: índice {topk_indices}")
    else:
        # Semântica via FAISS (sempre retorna índices dos blocos!)
        topk_indices = consultar_faiss(pergunta, k=8)
        log(f"Node: consultar_memoria | Semântica FAISS: {len(topk_indices)} blocos.")

    return {**state, "topk_indices": topk_indices}


def build_chat_prompt(turnos, pergunta_atual, modelo_format):
    """
    Gera prompt correto conforme o formato do modelo (chat ou completion).
    """
    if modelo_format == "openai" or modelo_format == "chat":
        mensagens = []
        for t in turnos:
            if t.get("pergunta"):
                mensagens.append({"role": "user", "content": t["pergunta"]})
            if t.get("resposta"):
                mensagens.append({"role": "assistant", "content": t["resposta"]})
        mensagens.append({"role": "user", "content": pergunta_atual})
        return mensagens
    else:
        # Para modelos completion (string única)
        prompt = ""
        for t in turnos:
            if t.get("pergunta"):
                prompt += f"Usuário: {t['pergunta']}\n"
            if t.get("resposta"):
                prompt += f"Agente: {t['resposta']}\n"
        prompt += f"Usuário: {pergunta_atual}\nAgente:"
        return prompt



def node_gerar_resposta_com_memoria(state: NexusState) -> NexusState:
    pergunta = state["pergunta"]
    session_id = state.get("session_id", "default")
    turnos = get_turnos(store, session_id)[-8:]

    model_name = state.get("model_activity_name", NODE_MODELS_DEFAULT["activity"]["name"])
    model_format = state.get("model_activity_format", NODE_MODELS_DEFAULT["activity"]["format"])

    # Adiciona o log para este nó específico
    log(f"Node: gerar_resposta_com_memoria | Usando modelo: {model_name} ({model_format})")

    # Prompt histórico de chat adaptável ao modelo
    user_prompt = build_chat_prompt(turnos, pergunta, model_format)

    # RAG: contexto adicional (atividade recente)
    topk_indices = state.get("topk_indices", [])
    memoria_util = get_blocks_by_indices(topk_indices) if topk_indices else []
    if memoria_util:
        context_rag = "\n".join(
            f"{i+1}. [{b.get('timestamp','')}] {b.get('context_summary','')[:500]} (Janela: {b.get('active_window','')})"
            for i, b in enumerate(memoria_util)
        )
        system_prompt = (
            "Você é um agente inteligente que monitora as atividades do computador do usuário. "
            "Você TEM acesso ao histórico de atividades, incluindo janelas abertas, OCR de tela, textos digitados e registros recentes. "
            "Seu papel é responder com base nesse contexto e no histórico do chat, inferindo sempre que possível. "
            "Nunca diga que não sabe, a não ser que realmente não haja nenhuma informação. "
            f"Aqui está o contexto adicional das atividades recentes:\n{context_rag}\nUtilize se relevante."
        )
    else:
        system_prompt = (
            "Você é um agente inteligente que monitora as atividades do computador do usuário. "
            "Mesmo sem dados recentes, responda com base no histórico do chat ou explique ao usuário o que está faltando."
        )

    response_data = ask_with_model(
        model_name,      # Usa a variável dinâmica
        model_format,    # Usa a variável dinâmica
        user_prompt=user_prompt,
        system_prompt=system_prompt
    )
    # return {**state, "resposta": resposta.content if hasattr(resposta, "content") else resposta}
    resposta_bruta = response_data.get("content", "Não foi possível gerar uma resposta.")
    duration = response_data.get("duration", 0)
    current_total_time = state.get("total_processing_time", 0.0)
    new_total_time = current_total_time + duration

    # Formata a resposta final para incluir o tempo de processamento
    resposta_formatada = f"{resposta_bruta}\n\n*— Modelo: {model_name} | Tempo Total: {new_total_time:.2f} segundos*"
    
    # Atualiza o estado com a resposta formatada e a duração
    return {**state, "resposta": resposta_formatada, "total_processing_time": new_total_time, "last_processing_time": duration}


def node_memoria(state: NexusState) -> NexusState:
    session_id = state.get("session_id", "default")
    turnos = get_turnos(store, session_id)
    pergunta = state.get("pergunta", "")
    resposta_completa = state.get("resposta", "")
    if pergunta and resposta_completa and "Ocorreu um erro" not in resposta_completa:
        resposta_limpa = resposta_completa.split("\n\n*— Tempo de processamento:")[0]
        
        novo_turno = {
            "pergunta": pergunta,
            "resposta": resposta_limpa, # Usa a resposta limpa
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        turnos.append(novo_turno)
        turnos = turnos[-10:]
        set_turnos(store, session_id, turnos)
        save_turnos_to_disk(store)
        log(f"Node: memoria | Turno salvo para a sessão {session_id}.")
        
    return {**state}
def node_resposta_direta(state: NexusState) -> NexusState:
    pergunta = state["pergunta"]
    
    model_name = state.get("model_principal_name", NODE_MODELS_DEFAULT["principal"]["name"])
    model_format = state.get("model_principal_format", NODE_MODELS_DEFAULT["principal"]["format"])
    duration = 0 # Inicializa a duração caso o modelo não seja chamado.
    log(f"Node: resposta_direta | Usando modelo: {model_name} ({model_format}) para pergunta: '{pergunta}'")
    
    # --- CORREÇÃO APLICADA ---
    response_data = ask_with_model(
        model_name=model_name,
        model_format=model_format,
        user_prompt=pergunta,
        system_prompt="Você é um assistente prestativo. Responda à pergunta do usuário de forma direta."
    )
    
    resposta_bruta = response_data.get("content", "")
    duration = response_data.get("duration", 0)
    current_total_time = state.get("total_processing_time", 0.0)
    new_total_time = current_total_time + duration

    resposta_formatada = f"{resposta_bruta}\n\n*— Modelo: {model_name} | Tempo Total: {new_total_time:.2f} segundos*"
    
    historico_fluxo.append({"nodo": "resposta_direta", "inicio": time.time(), "resposta": resposta_formatada})
    return {**state, "resposta": resposta_formatada, "total_processing_time": new_total_time, "last_processing_time": duration}

def node_executar_tool(state: NexusState) -> NexusState:
    pergunta = state["pergunta"]
    # Inicializa ou incrementa o contador de tentativas
    tentativas = state.get("tentativas", 0) + 1
    
    # Limite para evitar loop infinito
    if tentativas > 2:
        resposta = "A ferramenta falhou múltiplas vezes. Por favor, reformule sua pergunta ou tente um comando diferente."
        return {**state, "resposta": resposta}

    log(f"Node: executar_tool | Tentativa {tentativas} de executar tools com a pergunta: '{pergunta}'")
    
    try:
        if not pergunta:
            raise ValueError("Nenhuma pergunta recebida para execução de ferramenta.")

        # O ideal é que a invocação da ferramenta possa lançar uma exceção em caso de falha
        resposta_obj = tool_graph_executor.invoke({"input": pergunta})
        
        # Verifique se a saída indica um erro (alguns tools podem retornar texto de erro em vez de falhar)
        resposta = resposta_obj.get("output", str(resposta_obj))
        if "error" in resposta.lower() or "não encontrado" in resposta.lower():
             raise RuntimeError(f"A ferramenta retornou uma mensagem de erro: {resposta}")

        historico_fluxo.append({"nodo": "executar_tool", "inicio": time.time(), "resposta": resposta})
        # Sucesso: limpa os detalhes de erro e retorna a resposta
        return {**state, "resposta": resposta, "detalhes": None}

    except Exception as e:
        # FALHA: Captura a exceção e prepara o estado para o nó de correção
        log(f"ERRO no Node: executar_tool | Erro: {e}")
        error_message = f"Erro ao executar a ferramenta: {e}"
        historico_fluxo.append({"nodo": "executar_tool (FALHA)", "inicio": time.time(), "erro": error_message})
        
        # Atualiza o estado com as informações do erro para o próximo nó analisar
        return {**state, "detalhes": error_message, "tentativas": tentativas}


def node_interpretar_codigo(state: NexusState) -> NexusState:
    """
    Analisa uma pergunta sobre código, usando um resumo recente como contexto.
    """
    pergunta = state["pergunta"]
    log(f"Node: interpretar_codigo | Analisando pergunta sobre código: '{pergunta[:80]}...'")

    summary_code = load_latest_code_summary()
    resposta_bruta = ""
    duration = 0 

    if not summary_code or summary_code == "não encontrado":
        # Lógica sem contexto: informa o usuário que não há código para analisar.
        # Isso evita que o modelo alucine sobre um código que não viu.
        resposta = "Não posso responder a essa pergunta com precisão pois não tenho acesso a um resumo recente do seu código. Por favor, faça uma pergunta de programação mais genérica ou gere um novo resumo."
        resposta_obj = None # Para consistência com o fluxo abaixo

    else:
        # Lógica com contexto (RAG)
        agent_instructions = """
        Você é um assistente especialista em programação. Sua tarefa é analisar a pergunta de um usuário e o resumo de código fornecido para dar uma resposta técnica, clara e objetiva.

        - Analise o resumo de código dentro das tags <resumo_codigo> para entender o contexto do projeto.
        - Responda à pergunta do usuário, que está dentro das tags <pergunta>.
        - Se encontrar erros, explique-os. Se houver padrões de código ruins, sugira melhorias.
        - Se a pergunta for sobre funcionalidades, explique com base no código.
        - Responda como um desenvolvedor sênior: seja direto, preciso e baseie-se sempre no contexto fornecido.
        - Nunca dê respostas genéricas.
        """

        user_content = f"""
        <pergunta>
        {pergunta}
        </pergunta>

        <resumo_codigo>
        {summary_code}
        </resumo_codigo>
        """
        
        user_message_list = [{"role": "user", "content": user_content}]

        model_name = state.get("model_code_name", NODE_MODELS_DEFAULT["code"]["name"])
        model_format = state.get("model_code_format", NODE_MODELS_DEFAULT["code"]["format"])       
        log(f"Node: node_interpretar_codigo | Usando modelo: {model_name} ({model_format})")
        
        response_data = ask_with_model(
            model_name,
            model_format,        
            user_prompt=user_message_list,
            system_prompt=agent_instructions
        )

        resposta_bruta =  response_data.get("content", "Houve um problema ao gerar a resposta.")
        duration = response_data.get("duration", 0)
        current_total_time = state.get("total_processing_time", 0.0)
        new_total_time = current_total_time + duration

        # Formata a resposta final em ambos os casos (com ou sem contexto de código)
    resposta_formatada = f"{resposta_bruta}\n\n*— Modelo: {model_name} | Tempo Total: {new_total_time:.2f} segundos*"
    
    log("Node: interpretar_codigo | Resposta gerada.")
    historico_fluxo.append({"nodo": "interpretar_codigo", "inicio": time.time(), "resposta": resposta_formatada})
    
    return {**state, "resposta": resposta_formatada, "total_processing_time": new_total_time, "last_processing_time": duration}

def node_analisar_falha(state: NexusState) -> NexusState:
    """Analisa um erro ocorrido e formula uma resposta para o usuário."""
    log("Node: analisar_falha | Uma falha foi detectada. Analisando o erro.")
    
    pergunta_original = state["pergunta"]
    detalhes_erro = state["detalhes"]

    agent_instructions = """
    Você é um assistente de depuração. Uma tarefa que você tentou executar falhou.
    Seu objetivo é comunicar a falha ao usuário de forma clara e útil.
    Analise a pergunta original do usuário e a mensagem de erro que o sistema encontrou.

    - Explique o problema de forma simples.
    - Se possível, sugira como o usuário pode reformular a pergunta para ter sucesso.
    - Não invente uma resposta para a pergunta original. Apenas relate a falha.
    """

    user_content = f"""
    A pergunta original do usuário foi:
    <pergunta>{pergunta_original}</pergunta>

    A execução da ferramenta resultou no seguinte erro:
    <erro>{detalhes_erro}</erro>

    Por favor, formule uma resposta para o usuário explicando o que aconteceu.
    """
    
    user_message_list = [{"role": "user", "content": user_content}]

    model_name = state.get("model_principal_name", NODE_MODELS_DEFAULT["principal"]["name"])
    model_format = state.get("model_principal_format", NODE_MODELS_DEFAULT["principal"]["format"])      
    log(f"Node: node_analisar_falha | Usando modelo: {model_name} ({model_format})")
    # Usamos o modelo principal para "pensar" sobre o erro
    response_data = ask_with_model(
        model_name,
        model_format,
        user_prompt=user_message_list,
        system_prompt=agent_instructions
    )
    
    resposta_bruta = response_data.get("content", "")
    duration = response_data.get("duration", 0)
    current_total_time = state.get("total_processing_time", 0.0)
    new_total_time = current_total_time + duration

    resposta_formatada = f"{resposta_bruta}\n\n*— Modelo (Análise de Falha): {model_name} | Tempo Total: {current_total_time:.2f} segundos*"
    
    return {**state, "resposta": resposta_formatada, "total_processing_time": new_total_time, "last_processing_time": duration}

def rota_apos_tool(state: NexusState) -> str:
    """Decide se a execução da ferramenta foi bem-sucedida ou precisa de correção."""
    if state.get("detalhes") and "Erro ao executar" in state.get("detalhes", ""):
        log("Roteamento: Falha na ferramenta detectada. Indo para 'analisar_falha'.")
        return "analisar_falha"
    else:
        log("Roteamento: Ferramenta executada com sucesso. Indo para 'memoria'.")
        return "memoria"


def node_final(state: NexusState) -> dict:
    log("Node: finalizar | Fluxo concluído com sucesso.")
    historico_fluxo.append({"nodo": "finalizar", "inicio": time.time(), "fim": time.time()})
    os.makedirs("data/logs", exist_ok=True)
    with open("data/logs/fluxo_execucao.json", "w", encoding="utf-8") as f:
        json.dump(historico_fluxo, f, indent=2, ensure_ascii=False)
    return {}

# ========== WORKFLOW ==========

# Inicializa o grafo de estados
workflow = StateGraph(NexusState)

# Registro de todos os nós no grafo
workflow.add_node("processar_query", node_processar_query)
workflow.add_node("interpretador", node_interpretador)
workflow.add_node("consultar_memoria", node_consultar_memoria)
workflow.add_node("gerar_resposta_com_memoria", node_gerar_resposta_com_memoria)
workflow.add_node("interpretar_codigo", node_interpretar_codigo)
workflow.add_node("executar_tool", node_executar_tool)
workflow.add_node("analisar_falha", node_analisar_falha) # Nó de autocorreção
workflow.add_node("resposta_direta", node_resposta_direta)
workflow.add_node("memoria", node_memoria)
workflow.add_node("finalizar", node_final)

# Definição do ponto de entrada do grafo
workflow.set_entry_point("processar_query")

# Definição das arestas e do fluxo principal
workflow.add_edge("processar_query", "interpretador")

# Roteamento principal baseado na intenção detectada pelo interpretador
workflow.add_conditional_edges(
    "interpretador",
    lambda state: state.get("tipo", "fallback"),
    {
        # Caminhos que usam RAG (Recuperação de Informação)
        "temporal": "consultar_memoria",
        "mais_recente": "consultar_memoria",
        "semantica": "consultar_memoria",
        
        # Caminhos para nós especialistas
        "codigo": "interpretar_codigo",
        "tools": "executar_tool",
        "fallback": "resposta_direta",
    }
)

# Fluxo para respostas baseadas em memória (RAG)
workflow.add_edge("consultar_memoria", "gerar_resposta_com_memoria")
workflow.add_edge("gerar_resposta_com_memoria", "memoria")

# Fluxo do ciclo de autocorreção para ferramentas
workflow.add_conditional_edges("executar_tool", rota_apos_tool)
workflow.add_edge("analisar_falha", "memoria") # Após analisar a falha, salva a explicação na memória

# Fluxos que vão para a finalização
# Todos os caminhos que geram uma resposta final para o usuário convergem para o nó 'memoria'
# para garantir que a interação seja salva antes de finalizar.
workflow.add_edge("interpretar_codigo", "memoria")
workflow.add_edge("resposta_direta", "memoria")

# O nó 'memoria' é o último passo antes de encerrar o fluxo.
workflow.add_edge("memoria", "finalizar")

# Definição do ponto de saída do grafo
workflow.add_edge("finalizar", END)

# Compilação final do grafo
graph_executor = workflow.compile()
