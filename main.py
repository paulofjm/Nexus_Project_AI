import os
import sys
import time
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.window_tracker import track_activity
from nodes_graph.langgraph_nodes import graph_executor
from models.llm_manager import load_llm
from models.llm_config import NODE_MODELS_DEFAULT
warnings.filterwarnings("ignore", category=FutureWarning)  # Ajustar futuramente

# Carregamento dos modelos necessários
# load_llm(MODEL_CODE, MODEL_CODE_FORMAT)
# load_llm(MODEL_ACTIVITY, MODEL_ACTIVITY_FORMAT)
# load_llm(MODEL_WINDOW, MODEL_WINDOW_FORMAT)
# load_llm(MODEL_PRINCIPAL, MODEL_PRINCIPAL_FORMAT)
# load_llm(MODEL_ROUTER, MODEL_ROUTER_FORMAT)


def load_all_models(status_callback=None):
    """
    Carrega todos os modelos necessários para a aplicação.
    O callback é uma função que recebe uma string de status.
    """
    models_to_load = NODE_MODELS_DEFAULT
    
    total_models = len(models_to_load)
    for i, (name, (model_name, model_format)) in enumerate(models_to_load.items()):
        if status_callback:
            status_callback(f"Carregando modelo {name} ({model_name})...")
        load_llm(model_name, model_format)
        # Retorna o progresso como uma tupla (progresso_atual, progresso_total)
        yield (i + 1, total_models)




def thinking_animation(stop_event):
    animation = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        print(f"\r⏳ Processando... {animation[idx % len(animation)]}", end="")
        idx += 1
        time.sleep(0.2)

def continuous_tracking(stop_event):
    while not stop_event.is_set():
        track_activity()
        time.sleep(5)  # intervalo de captura


def run_interactive():
    print("\n🤖 Agente de Perguntas iniciado!\n")

    import threading
    import time

    from core.window_tracker import track_activity
    # from utils import code_monitor
    from nodes_graph.langgraph_nodes import graph_executor

    # (as outras importações e inicializações do seu main.py continuam...)

    # Thread para monitorar alterações em código
    # monitor_thread = threading.Thread(target=code_monitor.iniciar_monitoramento, daemon=True)
    # monitor_thread.start()

    # Thread para registrar atividades
    tracker_stop = threading.Event()
    tracker_thread = threading.Thread(target=continuous_tracking, args=(tracker_stop,), daemon=True)
    tracker_thread.start()

    try:
        while True:
            pergunta = input("\n🤖 Digite sua pergunta (ou 'sair'): ").strip()
            if pergunta.lower() in ["sair", "exit", "quit"]:
                print("👋 Encerrando...")
                break
            if not pergunta:
                print("⚠️ Pergunta vazia. Tente novamente.")
                continue

            stop_event = threading.Event()
            loading_thread = threading.Thread(target=thinking_animation, args=(stop_event,))
            loading_thread.start()

            try:
                final_state = graph_executor.invoke({"pergunta": pergunta})
            except Exception as e:
                final_state = {"resposta": f"⚠️ Erro na execução do grafo: {e}"}
            finally:
                stop_event.set()
                loading_thread.join()

            resposta_texto = final_state.get("resposta", "Nenhuma resposta gerada.")
            print("\n\n🧐 Resposta do Agente:")
            print(resposta_texto)
            print("-" * 80)
    finally:
        tracker_stop.set()
        tracker_thread.join()


if __name__ == "__main__":
    print("Carregando modelos para execução interativa...")
    # Itera sobre o gerador para carregar todos os modelos
    for _ in load_all_models(status_callback=print):
        pass
    run_interactive()