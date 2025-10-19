import re
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import ImageGrab, ImageChops, ImageStat, Image
from pynput.mouse import Controller
import pytesseract
from datetime import datetime
from core.storage import SCREENSHOT_DIR
from utils.logger import write_monitor_log as log
from dotenv import load_dotenv
import re
import mss
# from nltk.corpus import stopwords
# stopwords_all = set(stopwords.words('portuguese') + stopwords.words('english') + stopwords.words('spanish'))

load_dotenv()

tesseract_path = os.getenv("TESSERACT_PATH")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path



def maintain_latest_screenshots(directory: str, max_files: int = 20):
    """
    Mantém um número máximo de screenshots em um diretório, apagando os mais antigos.

    Args:
        directory (str): O caminho para a pasta de screenshots.
        max_files (int): O número máximo de arquivos a serem mantidos.
    """
    try:
        # Pega todos os arquivos .png no diretório
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".png") and os.path.isfile(os.path.join(directory, f))
        ]

        # Se o número de arquivos exceder o limite
        if len(files) > max_files:
            # Ordena os arquivos por tempo de modificação (os mais antigos primeiro)
            files.sort(key=os.path.getmtime)
            
            # Calcula quantos arquivos precisam ser apagados
            files_to_delete = files[:len(files) - max_files]
            
            # log(f"Limpeza: {len(files)} screenshots encontrados. Apagando {len(files_to_delete)} mais antigos para manter o limite de {max_files}.")

            # Apaga os arquivos mais antigos
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except OSError as e:
                    log(f"❌ ERRO: Não foi possível apagar o arquivo {file_path}. Erro: {e}")
                    
    except FileNotFoundError:
        log(f"⚠️ AVISO: O diretório de screenshots '{directory}' não foi encontrado.")
    except Exception as e:
        log(f"❌ ERRO inesperado na função de limpeza de screenshots: {e}")

def filtro_linguistico(ocr_text):
    """
    Filtra o texto do OCR, removendo linhas vazias e duplicadas.
    """
    # 1. Pega todas as linhas do texto que não estão em branco
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    # 2. Usa um conjunto para rastrear linhas únicas e evitar duplicatas
    seen = set()
    filtered_lines = []
    
    # 3. Itera sobre as linhas que contêm texto
    for line in lines:
        # Normaliza a linha para a verificação de duplicatas
        normalized_line = line.lower()
        if normalized_line not in seen:
            seen.add(normalized_line)
            filtered_lines.append(line)
        
        # Limita a saída para as primeiras 20 linhas únicas
        if len(filtered_lines) >= 20:
            break
            
    # 4. Junta as linhas filtradas e limita o tamanho total
    return "\n".join(filtered_lines)[:1500]

def extract_text_from_image(image_path):
    # Garante que SCREENSHOT_DIR existe
    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)
        
    try:
        img = Image.open(image_path)
        raw_text = pytesseract.image_to_string(img, lang="por+eng+spa")
        clean_text = raw_text.strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)
        # Agora o filtro funcionará
        clean_text = filtro_linguistico(clean_text)
        return clean_text
    except Exception as e:
        # print(f"Erro no OCR: {e}") # Usando print para debug imediato
        log(f"Erro no OCR: {e}")
        return ""


def movimento_detectado(img1, img2, threshold=10):
    diff = ImageChops.difference(img1, img2)
    stat = ImageStat.Stat(diff)
    rms = sum([v ** 2 for v in stat.mean]) ** 0.5
    return rms > threshold



def capture_full_screenshot_with_motion(
    output_dir,
    ultima_imgs=None,
    ultima_pos=None,
    last_capture_time=0,
    interval=5,
    threshold=10
):
    os.makedirs(output_dir, exist_ok=True)
    screenshots = []
    now = time.time()
    if now - last_capture_time < interval:
        return [], None

    mouse = Controller()
    current_pos = mouse.position

    movimento_mouse = (ultima_pos is None) or (current_pos != ultima_pos)

    with mss.mss() as sct:
        # CORREÇÃO: Lógica para selecionar o(s) monitor(es) correto(s)
        monitors = sct.monitors[1:]  # Começa com o padrão de monitores secundários
        # Se a lista acima estiver vazia (cenário de monitor único), usa o monitor primário
        if not monitors:
            monitors = [sct.monitors[1]] # Usa apenas o primário em uma lista

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        imgs_pil = []
        
        # O loop agora funciona para múltiplos ou um único monitor
        for idx, monitor in enumerate(monitors):
            filename = f"screenshot_monitor{idx+1}_{timestamp}.png"
            path = os.path.join(output_dir, filename)
            img_mss = sct.grab(monitor)
            mss.tools.to_png(img_mss.rgb, img_mss.size, output=path)
            img_pil = Image.frombytes("RGB", img_mss.size, img_mss.rgb)
            imgs_pil.append(img_pil)

            movimento_tela = True if not ultima_imgs or idx >= len(ultima_imgs) else movimento_detectado(ultima_imgs[idx], img_pil, threshold)

            if movimento_tela or movimento_mouse:
                screenshots.append(path)
            else:
                if os.path.exists(path):
                    os.remove(path)

    last_valid_path = screenshots[-1] if screenshots else None
    return screenshots, last_valid_path