import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import psutil
import pygetwindow as gw
from datetime import datetime
import pyautogui
from utils.ocr import extract_text_from_image, capture_full_screenshot_with_motion, maintain_latest_screenshots
from core.context import resumir_arquivo_compilado, generate_compiled_code #,get_project_files_summary
from core.storage import SCREENSHOT_DIR, detect_active_project_root
from utils.logger import write_monitor_log as log
import hashlib
from utils.metadata_compactor import (
    salvar_em_buffer,
    compactar_bloco_de_20,
    consolidar_blocos_medios,
    init_faiss_if_missing
)

from models.llm_manager import ask_with_model
from models.llm_config import NODE_MODELS_DEFAULT



# Estado anterior
last_mouse_pos = None
last_window_title = None
last_screenshot_hash = None

# Função para verificar se houve atividade relevante
def houve_atividade_relevante(duracao=3, intervalo=0.5, min_atividades=2):
    """
    Verifica se houve pelo menos `min_atividades` diferentes (mouse, janela ou visual)
    ao longo de `duracao` segundos, em janelas de `intervalo` segundos.
    """
    global last_mouse_pos, last_window_title, last_screenshot_hash
    contador = 0

    for _ in range(int(duracao / intervalo)):
        atividade = False

        # Verifica posição do mouse
        current_mouse_pos = pyautogui.position()
        if current_mouse_pos != last_mouse_pos:
            atividade = True
        last_mouse_pos = current_mouse_pos

        # Verifica mudança de janela
        current_window = gw.getActiveWindow()
        current_title = current_window.title if current_window else "Unknown"
        if current_title != last_window_title:
            atividade = True
        last_window_title = current_title

        # Verifica mudança visual (screenshot hash reduzido)
        screenshot = pyautogui.screenshot().resize((100, 100))
        screenshot_hash = hashlib.sha256(screenshot.tobytes()).hexdigest()
        if screenshot_hash != last_screenshot_hash:
            atividade = True
        last_screenshot_hash = screenshot_hash

        if atividade:
            contador += 1

        time.sleep(intervalo)

    return contador >= min_atividades

# Função principal para rastrear a atividade da janela e capturar screenshots
def track_activity():
    if not houve_atividade_relevante():
        log("⏸️ Nenhuma atividade detectada de forma contínua. Registro ignorado.")
        return 
          
    init_faiss_if_missing()
    active_window = gw.getActiveWindow()
    active_title = active_window.title if active_window else "Unknown"
    ultima_img = None
    ultima_pos = None
    
    last_capture_time = 0
    INTERVAL = 5  # segundos entre prints

    # screenshots = capture_full_screenshot_with_motion(
    #     SCREENSHOT_DIR, ultima_img, ultima_pos, last_capture_time, interval=INTERVAL
    # )
    screenshots, last_screenshot = capture_full_screenshot_with_motion(
        SCREENSHOT_DIR, ultima_img, ultima_pos, last_capture_time, interval=INTERVAL
    )
    # screenshots é uma lista (pode ser vazia, 1, ou muitos caminhos)
    if screenshots:
        all_extracted_text = []
        for screenshot_path in screenshots:
            ocr_text_part = extract_text_from_image(screenshot_path) if screenshot_path else ""
            if ocr_text_part:
                all_extracted_text.append(ocr_text_part)

        # 2. Junte todo o texto extraído em uma única variável
        # O separador "\n--- FIM DO SCREENSHOT ---\n" ajuda a manter o contexto separado
        final_ocr_text = "\n--- FIM DO SCREENSHOT ---\n".join(all_extracted_text)

        # 3. Use a variável final com todo o texto para o resto do processamento
        if final_ocr_text:
            cpu_usage = psutil.cpu_percent(interval=0.5)
            memory_usage = psutil.virtual_memory().percent
            # summary_path = generate_compiled_code(os.path.abspath(os.path.join(BASE_DIR, "..")))
            project_root = detect_active_project_root()
            summary_path = generate_compiled_code(project_root)
            # code_summary = resumir_arquivo_compilado()

            #    Isso diz ao modelo QUAL É o seu papel.
            agent_instructions = """
            Você é um agente especialista em analisar textos brutos de OCR (Reconhecimento Óptico de Caracteres) extraídos da tela de um computador.
            Seu único papel é encontrar palavras-chave e resumir objetivamente o que o usuário está fazendo, com base nos textos fornecidos.
            Os textos podem parecer sem sentido ou conter palavras soltas, mas você deve fazer o seu melhor para contextualizá-los.
            NÃO é um chat. NÃO forneça ajuda, sugestões, saudações ou qualquer texto adicional.
            Apenas forneça o resumo conciso da atividade.
            """.strip()

            #    Isto é O QUE o modelo deve analisar.
            user_content = f"""
            Textos extraídos do OCR:
            {final_ocr_text}
            """.strip()

            # 3. Formate a mensagem do usuário como a função espera para o formato "openai".
            user_message_list = [{"role": "user", "content": user_content}]


            response_data  = ask_with_model(
                model_name=NODE_MODELS_DEFAULT["activity"]["name"],
                model_format=NODE_MODELS_DEFAULT["activity"]["format"],
                user_prompt=user_message_list,  # Passando a lista de mensagens corretamente
                system_prompt=agent_instructions # Passando as instruções detalhadas no parâmetro correto
            )
            
            # if hasattr(context_summary, "content"):
            #     context_summary = context_summary.content
            context_summary = response_data.get("content", "Não foi possível gerar um resumo da atividade.")

            # Salva um único registro com o conteúdo de todas as telas
            # Você pode querer decidir qual screenshot salvar (ex: o último)
            # last_screenshot = screenshots[-1] if screenshots else ""

            record = {
                    "timestamp": datetime.now().isoformat(),
                    "active_window": active_title,
                    "screenshot": last_screenshot.replace("\\", "/") if last_screenshot else "",
                    "extracted_text": final_ocr_text[:300], # Mantem apenas os 300 primeiros caracteres para verificação, o resumo é passado no contexto_summary
                    "typed_text": "",
                    "cpu_percent": cpu_usage,
                    "memory_percent": memory_usage,
                    "context_summary": context_summary,
                    "code_version_path": summary_path,
                    "nivel": "curto",
                    # As tags agora funcionam com o texto de todas as telas
                    "tags": list(set([
                        "navegador" if any(x in active_title.lower() for x in ["chrome", "edge", "firefox"]) else "",
                        "desenvolvimento" if any(x in final_ocr_text.lower() for x in ["stack", "git", "python", "error"]) else "",
                        "documento" if any(x in final_ocr_text.lower() for x in ["word", ".doc", ".pdf", "texto"]) else ""
                    ]) - {""}),
                    # "project_files_summary": code_summary
                    # "project_code_summary_file": summary_path
                }
            
            salvar_em_buffer(record)
            compactar_bloco_de_20()
            consolidar_blocos_medios()

            maintain_latest_screenshots(SCREENSHOT_DIR)
        else:
            time.sleep(0.5)