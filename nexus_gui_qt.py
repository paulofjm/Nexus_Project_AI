import sys
import os
import threading
import uuid
import json
import markdown
import subprocess
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QComboBox, QStackedWidget,
                             QScrollArea, QFrame, QListWidget, QListWidgetItem, QMenu,
                             QMessageBox)
from PySide6.QtGui import QMovie, QIcon, QPixmap, QAction
from PySide6.QtCore import Signal, QObject, Qt, QThread
# from models.llm_config import AVAILABLE_MODELS, set_selected_model
from models.llm_manager import check_api_key
from main import continuous_tracking
from core.storage import CHAT_TITLES_FILE, ASSETS_DIR, BASE_DIR
from nodes_graph.langgraph_nodes import graph_executor, store, load_turnos_from_disk, get_turnos, save_turnos_to_disk
from models.llm_config import AVAILABLE_MODELS, NODE_MODELS_DEFAULT, get_format_from_name
# Mapeamento reverso para obter o nome amigável a partir do ID do modelo
AVAILABLE_MODELS_REVERSE = {v: k for k, v in AVAILABLE_MODELS.items()}

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(PROJECT_ROOT)

# 3. Usa a PROJECT_ROOT para definir os caminhos dos arquivos de dados.
# DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# ASSETS_DIR = os.path.join(PROJECT_ROOT, 'gui', 'assets') # Caminho para os assets

# Garante que o diretório de dados exista
# os.makedirs(DATA_DIR, exist_ok=True)


# --- Classes de Worker e Sinais ---
class WorkerSignals(QObject):
    update_chat = Signal(str, str)
    finished = Signal()

class ChatWorker(QObject):
    def __init__(self, graph_input: dict):
        super().__init__()
        self.graph_input = graph_input # Armazena o dicionário completo
        self.signals = WorkerSignals()

    def run(self):
        try:
            # Usa o dicionário de entrada diretamente ao chamar o grafo
            response_generator = graph_executor.stream(self.graph_input)
            final_answer = ""
            for response_chunk in response_generator:
                node_name = list(response_chunk.keys())[0]
                node_output = response_chunk[node_name]
                if node_name in ["gerar_resposta_com_memoria", "resposta_direta", "interpretar_codigo", "executar_tool", "analisar_falha"]:
                    if 'resposta' in node_output:
                        final_answer = node_output['resposta']
                        self.signals.update_chat.emit("assistant", final_answer)
        except Exception as e:
            error_message = f"Ocorreu um erro: {e}"
            self.signals.update_chat.emit("assistant", error_message)
        finally:
            self.signals.finished.emit()

# --- Classe Principal da Aplicação ---
class NexusApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_session_id = None
        self.session_titles = {}
        self.chat_history_layout = None
        self.scroll_area = None
        self.worker_thread = None
        self.chat_worker = None
        self.tracker_thread = None
        self.tracker_stop = threading.Event()
        self.model_selections = {} # Armazena as seleções de modelo da GUI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Nexus Agent")
        self.setGeometry(100, 100, 900, 700)
        icon_path = os.path.join(ASSETS_DIR, "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        main_layout = QVBoxLayout(self)
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        self.create_setup_widget()
        self.create_chat_widget()

        self.stacked_widget.addWidget(self.setup_widget)
        self.stacked_widget.addWidget(self.chat_widget)

        self.apply_stylesheet()
        
        # self.model_combo.currentTextChanged.connect(self._on_model_changed)
        # self._on_model_changed(self.model_combo.currentText())

    def _update_api_key_visibility(self):
        """Verifica todos os seletores de modelo e mostra o campo de API se algum for da OpenAI."""
        uses_openai = False
        # Itera sobre os valores do dicionário que contém os QComboBox
        for combo in self.node_model_combos.values():
            if "openai" in combo.currentText().lower():
                uses_openai = True
                break  # Se um for encontrado, não precisa checar os outros

        # Mostra ou esconde o campo de API com base na verificação
        self.api_key_input.setVisible(uses_openai)
        if not uses_openai:
            self.api_key_input.clear() # Limpa a chave se não for necessária


    def create_setup_widget(self):
        self.setup_widget = QWidget()
        setup_layout = QVBoxLayout(self.setup_widget)
        setup_layout.setAlignment(Qt.AlignCenter)
        setup_layout.setSpacing(20)

        logo_label = QLabel()
        logo_path = os.path.join(ASSETS_DIR, "logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        setup_layout.addWidget(logo_label)
        
        setup_layout.addWidget(QLabel("Modelos para cada Agente Especialista:"))
        
        # Dicionário para guardar as comboboxes
        self.node_model_combos = {}

        # Nós a serem exibidos na GUI
        node_display_names = {
            "router": "Agente Roteador (Interpretador)",
            "activity": "Agente de Atividade (Memória/RAG)",
            "code": "Agente de Código",
            "principal": "Agente Principal (Fallback)"
        }

        for key, display_name in node_display_names.items():
            # Layout para cada linha de seleção de modelo
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{display_name}:"))
            
            combo = QComboBox()
            combo.addItems(list(AVAILABLE_MODELS.keys()))

            # Define o valor padrão
            default_model_name = NODE_MODELS_DEFAULT[key]["name"]
            if default_model_name in AVAILABLE_MODELS_REVERSE:
                combo.setCurrentText(AVAILABLE_MODELS_REVERSE[default_model_name])

            combo.currentTextChanged.connect(self._update_api_key_visibility)

            row_layout.addWidget(combo)
            setup_layout.addLayout(row_layout)
            self.node_model_combos[key] = combo # Guarda a referência da combobox

        setup_layout.addSpacing(20) # Espaçamento

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Insira sua chave de API da OpenAI (se necessário)")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        # self.api_key_input.textChanged.connect(self._on_model_changed)
        setup_layout.addWidget(self.api_key_input)
        
        # 1. Cria os botões
        # self.btn_avancar = QPushButton("Avançar")
        # self.btn_avancar.setEnabled(False)
        # self.btn_avancar.clicked.connect(self.go_to_chat)

        self.btn_avancar = QPushButton("Avançar")
        self.btn_avancar.clicked.connect(self.go_to_chat)
        setup_layout.addWidget(self.btn_avancar)

        self.btn_reset = QPushButton("Resetar Agente")
        self.btn_reset.clicked.connect(self.execute_reset_agent)
        self.btn_reset.setObjectName("ResetButton")

        # 2. Adiciona o botão "Avançar" diretamente ao layout vertical.
        #    Ele se expandirá para ocupar a largura disponível.
        setup_layout.addWidget(self.btn_avancar)

        # 3. Cria um layout horizontal separado para o botão de reset.
        reset_button_layout = QHBoxLayout()
        reset_button_layout.addStretch()  # Empurra o botão para a direita.
        reset_button_layout.addWidget(self.btn_reset)

        # 4. Adiciona o layout do botão de reset ao layout principal.
        setup_layout.addLayout(reset_button_layout)
        
        # 5. Atualiza a visibilidade do campo de chave API.
        self._update_api_key_visibility() 


    def create_chat_widget(self):
        self.chat_widget = QWidget()
        main_chat_layout = QHBoxLayout(self.chat_widget)
        
        sessions_panel = QWidget()
        sessions_panel.setFixedWidth(200)
        sessions_layout = QVBoxLayout(sessions_panel)
        
        self.btn_new_chat = QPushButton("+ Nova Conversa")
        self.btn_new_chat.clicked.connect(self.new_chat_session)
        sessions_layout.addWidget(self.btn_new_chat)
        
        self.session_list_widget = QListWidget()
        self.session_list_widget.currentItemChanged.connect(self.switch_session)
        self.session_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list_widget.customContextMenuRequested.connect(self.show_session_context_menu)
        self.session_list_widget.itemChanged.connect(self.on_session_renamed)
        sessions_layout.addWidget(self.session_list_widget)

        sessions_layout.addStretch()
        self.btn_inicio = QPushButton("Início")
        self.btn_inicio.clicked.connect(self.go_to_setup)
        sessions_layout.addWidget(self.btn_inicio)
        main_chat_layout.addWidget(sessions_panel)

        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        self.chat_history_container = QWidget()
        self.chat_history_layout = QVBoxLayout(self.chat_history_container)
        self.chat_history_layout.setAlignment(Qt.AlignTop)
        
        self.scroll_area.setWidget(self.chat_history_container)
        chat_layout.addWidget(self.scroll_area)

        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Digite sua pergunta...")
        self.user_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.user_input)

        self.send_button = QPushButton("Enviar")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        chat_layout.addLayout(input_layout)

        main_chat_layout.addWidget(chat_panel)

    def _on_model_changed(self, model_name=""):
        selected_model = self.model_combo.currentText()
        is_local = "gguf" in selected_model.lower() or "ollama" in selected_model.lower()

        if is_local:
            self.api_key_input.setPlaceholderText("Não é necessário para modelos locais")
            self.api_key_input.setEnabled(False)
            self.api_key_input.clear()
            self.btn_avancar.setEnabled(True)
        else:
            self.api_key_input.setPlaceholderText("Insira sua chave de API aqui")
            self.api_key_input.setEnabled(True)
            self.btn_avancar.setEnabled(len(self.api_key_input.text().strip()) > 0)


    def go_to_chat(self):
        # Limpa seleções antigas
        self.model_selections = {}
        
        api_key = self.api_key_input.text().strip()
        uses_openai = False

        # Itera sobre as comboboxes dos nós
        for key, combo in self.node_model_combos.items():
            friendly_name = combo.currentText()
            model_id = AVAILABLE_MODELS[friendly_name]
            model_format = get_format_from_name(model_id)

            # Determina o formato e armazena a seleção
            if "openai" in model_id:
                uses_openai = True
            # elif "ollama" in friendly_name.lower():
            #     model_format = "ollama"
            # elif "gguf" in friendly_name.lower():
            #     model_format = "gguf"
            # else:
            #     model_format = "transformers"

            self.model_selections[f"model_{key}_name"] = model_id
            self.model_selections[f"model_{key}_format"] = model_format

        # Verifica a chave da API apenas se um modelo da OpenAI for selecionado
        if uses_openai:
            if not check_api_key("openai", api_key):
                QMessageBox.warning(self, "Chave de API Inválida", "Você selecionou um modelo da OpenAI, mas a chave de API fornecida é inválida.")
                return
            # Define a chave globalmente para ser usada pelo `OpenAILLM`
            os.environ["OPENAI_API_KEY"] = api_key


        self.on_inicializacao_finalizada()


    def on_inicializacao_finalizada(self):
        self.stacked_widget.setCurrentWidget(self.chat_widget)
        if self.tracker_thread is None:
            self.tracker_thread = threading.Thread(target=continuous_tracking, args=(self.tracker_stop,), daemon=True)
            self.tracker_thread.start()
        
        self.load_sessions()
        if self.session_list_widget.count() == 0:
            self.new_chat_session()
        else:
            self.session_list_widget.setCurrentRow(0)

    def go_to_setup(self):
        self.clear_chat_display()
        self.stacked_widget.setCurrentWidget(self.setup_widget)

    def load_titles_from_disk(self):
        try:
            if os.path.exists(CHAT_TITLES_FILE):
                try:
                    with open(CHAT_TITLES_FILE, 'r', encoding='utf-8') as f:
                        self.session_titles = json.load(f)
                except (json.JSONDecodeError, IOError):
                    # Se o arquivo existir mas for inválido, usa um dict vazio
                    self.session_titles = {}
            else:
                # Se o arquivo não existir, também usa um dict vazio
                self.session_titles = {}
        except (json.JSONDecodeError, IOError):
            self.session_titles = {}

    def save_titles_to_disk(self):
        os.makedirs(os.path.dirname(CHAT_TITLES_FILE), exist_ok=True)
        with open(CHAT_TITLES_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.session_titles, f, indent=4)

    def load_sessions(self):
        self.load_titles_from_disk()
        self.session_list_widget.clear()
        load_turnos_from_disk(store)
        session_ids = list(store.yield_keys())
        for session_id in session_ids:
            session_title = self.session_titles.get(session_id)
            if not session_title:
                turnos = get_turnos(store, session_id)
                session_title = turnos[0]['pergunta'][:30] if turnos and turnos[0].get('pergunta') else f"Chat {session_id[:8]}"
            
            item = QListWidgetItem(session_title)
            item.setData(Qt.UserRole, session_id)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.session_list_widget.addItem(item)

    def new_chat_session(self):
        new_id = str(uuid.uuid4())
        self.current_session_id = new_id
        
        item = QListWidgetItem(f"Novo Chat...")
        item.setData(Qt.UserRole, new_id)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.session_list_widget.insertItem(0, item)
        self.session_list_widget.setCurrentItem(item)
        
        self.clear_chat_display()
        self.add_message_to_chat("assistant", "Olá! Sou o Nexus. Como posso ajudar nesta nova conversa?")

    def switch_session(self, current_item, previous_item=None):
        if current_item is None:
            return
        session_id = current_item.data(Qt.UserRole)
        self.current_session_id = session_id
        
        self.clear_chat_display()
        
        turnos = get_turnos(store, session_id)
        if not turnos:
            self.add_message_to_chat("assistant", "Continue esta conversa...")
        else:
            for turno in turnos:
                if turno.get("pergunta"):
                    self.add_message_to_chat("user", turno["pergunta"])
                if turno.get("resposta"):
                    self.add_message_to_chat("assistant", turno["resposta"])

    def show_session_context_menu(self, position):
        item = self.session_list_widget.itemAt(position)
        if item:
            menu = QMenu()
            
            rename_action = QAction("Renomear", self)
            rename_action.triggered.connect(lambda: self.rename_session(item))
            menu.addAction(rename_action)
            
            delete_action = QAction("Excluir", self)
            delete_action.triggered.connect(lambda: self.delete_session(item))
            menu.addAction(delete_action)

            menu.exec(self.session_list_widget.mapToGlobal(position))

    def rename_session(self, item):
        self.session_list_widget.editItem(item)

    def on_session_renamed(self, item):
        session_id = item.data(Qt.UserRole)
        new_title = item.text()
        self.session_titles[session_id] = new_title
        self.save_titles_to_disk()
    
    def delete_session(self, item):
        session_id = item.data(Qt.UserRole)
        
        reply = QMessageBox.question(self, 'Confirmar Exclusão', 
                                     f"Você tem certeza que deseja excluir a conversa '{item.text()}'?\nEsta ação não pode ser desfeita.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            store.mdelete([session_id])
            save_turnos_to_disk(store)
            
            if self.session_titles.pop(session_id, None):
                self.save_titles_to_disk()

            row = self.session_list_widget.row(item)
            self.session_list_widget.takeItem(row)

            if self.session_list_widget.count() == 0:
                self.new_chat_session()
            else:
                self.session_list_widget.setCurrentRow(0)

    def execute_reset_agent(self):
        reply = QMessageBox.warning(self, 'Confirmar Reset', 
                                     "Você tem certeza que deseja resetar o agente?\nTODOS os dados (memória, logs, conversas) serão apagados permanentemente.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, 
                                     QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Yes:
            # Constrói o caminho para o script reset_agent.py na pasta raiz
            script_path = os.path.join(BASE_DIR, "reset_agent.py")
            if not os.path.exists(script_path):
                QMessageBox.critical(self, "Erro", f"O script 'reset_agent.py' não foi encontrado na raiz do projeto.")
                return

            try:
                # Usa sys.executable para garantir que está usando o mesmo interpretador python
                subprocess.Popen([sys.executable, script_path])
                QMessageBox.information(self, "Sucesso", "O agente foi resetado. O aplicativo será fechado agora.")
                self.close()
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao executar o script de reset: {e}")

    def clear_chat_display(self):
        for i in reversed(range(self.chat_history_layout.count())): 
            widget_to_remove = self.chat_history_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

    def send_message(self):
        if not self.current_session_id:
            self.new_chat_session()

        user_text = self.user_input.text().strip()
        if user_text:
            self.add_message_to_chat("user", user_text)
            self.user_input.clear()
            
            current_item = self.session_list_widget.currentItem()
            if current_item and "Novo Chat" in current_item.text():
                new_title = user_text[:30] + "..."
                current_item.setText(new_title)
                self.on_session_renamed(current_item)

            self.process_with_langgraph(user_text)

    def add_message_to_chat(self, sender, text):
        message_label = QLabel()
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        message_label.setOpenExternalLinks(True)
        
        if sender == "user":
            message_label.setText(text)
            message_label.setAlignment(Qt.AlignRight)
            message_label.setStyleSheet("background-color: #2a3b4c; color: white; padding: 10px; border-radius: 10px; margin-left: 50px;")
        else:
            html_text = markdown.markdown(text, extensions=['fenced_code', 'tables'])
            message_label.setText(html_text)
            message_label.setAlignment(Qt.AlignLeft)
            message_label.setStyleSheet("background-color: #3b4a5a; color: white; padding: 10px; border-radius: 10px; margin-right: 50px;")
            
        self.chat_history_layout.addWidget(message_label)
        QApplication.processEvents()
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def process_with_langgraph(self, question):
        self.send_button.setEnabled(False)
        self.user_input.setEnabled(False)

        self.thinking_label = QLabel("Nexus está pensando...")
        loading_path = os.path.join(ASSETS_DIR, "loading.gif")
        if os.path.exists(loading_path):
            movie = QMovie(loading_path)
            self.thinking_label.setMovie(movie)
            movie.start()
        self.thinking_label.setStyleSheet("background-color: #3b4a5a; color: white; padding: 10px; border-radius: 10px; margin-right: 50px;")
        self.chat_history_layout.addWidget(self.thinking_label)

        graph_input = {
            "pergunta": question,
            "session_id": self.current_session_id,
            **self.model_selections  # 2. Desempacota as seleções de modelo aqui!
        }
        

        self.chat_worker = ChatWorker(graph_input)
        self.worker_thread = QThread()
        self.chat_worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.chat_worker.run)
        self.chat_worker.signals.finished.connect(self.worker_thread.quit)
        self.chat_worker.signals.finished.connect(self.chat_worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.chat_worker.signals.update_chat.connect(self.update_assistant_message)
        self.worker_thread.finished.connect(self.on_processing_finished)
    

        self.worker_thread.start()

    def update_assistant_message(self, sender, text):
        if hasattr(self, 'thinking_label') and self.thinking_label:
            self.thinking_label.setParent(None)
            self.thinking_label = None
        self.add_message_to_chat(sender, text)

    def on_processing_finished(self):
        if hasattr(self, 'thinking_label') and self.thinking_label and self.thinking_label.parent() is not None:
             self.thinking_label.setParent(None)
             self.thinking_label = None
        self.send_button.setEnabled(True)
        self.user_input.setEnabled(True)
        self.user_input.setFocus()
        
    def closeEvent(self, event):
        """
        Garante que o programa feche corretamente sem travar.
        """
        print("INFO: Recebido evento para fechar a janela.")
        # Sinaliza para o thread de monitoramento parar.
        self.tracker_stop.set()
        
        # Não usamos mais o .join() aqui, pois ele bloqueia a thread da GUI.
        # A flag 'daemon=True' no thread já garante que ele será encerrado
        # quando o programa principal terminar.
        
        print("INFO: Aplicação encerrando.")
        super().closeEvent(event)


    def apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e2a38;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QListWidget {
                background-color: #2a3b4c;
                border: 1px solid #3b4a5a;
                padding: 5px;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                background-color: #007acc;
                color: white;
            }
            QListWidget QLineEdit {
                background-color: #3b4a5a;
                color: #ffffff;
                border: 1px solid #007acc;
                padding: 2px;
                margin: 1px;
            }
            QLabel { font-size: 14px; }
            QLabel a {
                color: #8af;
                text-decoration: none;
            }
            QLineEdit {
                background-color: #2a3b4c;
                border: 1px solid #3b4a5a;
                padding: 8px; border-radius: 5px; font-size: 14px;
            }
            QPushButton {
                background-color: #007acc; color: white; border: none;
                padding: 10px 15px; border-radius: 5px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005f9e;
            }
            QPushButton:disabled {
                background-color: #555;
            }
            #ResetButton {
                background-color: #5a3b3b;
                max-width: 120px;
                font-size: 12px;
                font-weight: normal;
                padding: 8px 12px;
            }
            #ResetButton:hover {
                background-color: #8b4a4a;
            }
            QComboBox {
                background-color: #2a3b4c; border: 1px solid #3b4a5a;
                padding: 8px; border-radius: 5px; font-size: 14px;
            }
            QComboBox::drop-down { border: none; }
            QScrollArea { border: none; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = NexusApp()
    ex.show()
    sys.exit(app.exec())