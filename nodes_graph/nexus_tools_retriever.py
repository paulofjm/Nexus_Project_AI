from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
import os

from core.context import consultar_faiss
from utils.ocr import capture_full_screenshot_with_motion, extract_text_from_image
from models.llm_manager import load_llm
from langchain_community.tools import tool

# =============================
# TOOLS (todos com docstring)
# =============================

@tool
def listar_diretorio(path: str) -> str:
    """Lista os arquivos de um diretório. Argumento: caminho da pasta."""
    try:
        arquivos = os.listdir(path)
        return "\n".join(arquivos) if arquivos else "Diretório vazio."
    except Exception as e:
        return f"Erro ao listar diretório: {e}"

@tool
def ler_arquivo(path: str) -> str:
    """Lê e retorna o conteúdo de um arquivo de texto (até 3000 caracteres)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(3000)
    except Exception as e:
        return f"Erro ao abrir arquivo: {e}"

@tool
def tirar_screenshot(_: str = "") -> str:
    """Tira um screenshot da tela inteira e salva em arquivo temporário."""
    try:
        path, _ = capture_full_screenshot_with_motion()
        return f"Screenshot salva em: {path}"
    except Exception as e:
        return f"Erro ao capturar screenshot: {e}"

@tool
def ocr_da_tela(_: str = "") -> str:
    """Executa OCR (reconhecimento de texto) da tela atual."""
    try:
        path, _ = capture_full_screenshot_with_motion()
        texto = extract_text_from_image(path)
        return texto or "Nenhum texto encontrado."
    except Exception as e:
        return f"Erro ao extrair texto: {e}"

TOOLS_NEXUS = [listar_diretorio, ler_arquivo, tirar_screenshot, ocr_da_tela]

# =============================
# RETRIEVER (FAISS + LangChain)
# =============================
class NexusRetriever(BaseRetriever):
    k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        blocos = consultar_faiss(query, k=self.k)
        return [
            Document(
                page_content=b.get("context_summary", ""),
                metadata={
                    "timestamp": b.get("timestamp_inicio"),
                    "janela": b.get("janela_mais_frequente"),
                    "nivel": b.get("nivel", "medio"),
                    "score": b.get("score", 0.0)
                }
            ) for b in blocos
        ]

RETRIEVER_NEXUS = NexusRetriever()