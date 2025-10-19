import os
import glob
from utils.logger import write_monitor_log as log

# Caminho base do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data"))

def apagar_arquivos_data():
    extensoes = ["json", "faiss", "txt", "png", "jpg", "jpeg", "gif", "mp4", "webm", "wav", "mp3"]
    arquivos_para_apagar = []

    # Busca todos os arquivos com extensões relevantes dentro de /data e subpastas
    for ext in extensoes:
        arquivos_para_apagar.extend(glob.glob(os.path.join(DATA_DIR, "**", f"*.{ext}"), recursive=True))

    # Remove cada arquivo encontrado
    for arquivo in arquivos_para_apagar:
        try:
            os.remove(arquivo)
            print(f"🗑️ Apagado: {arquivo}")
            log(f"Arquivo removido: {arquivo}")
        except Exception as e:
            print(f"❌ Erro ao apagar {arquivo}: {e}")
            log(f"Erro ao apagar {arquivo}: {e}")

if __name__ == "__main__":
    print("🧹 Resetando arquivos em /data ...")
    apagar_arquivos_data()
    print("✅ Reset concluído.")
