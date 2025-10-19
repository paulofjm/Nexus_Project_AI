import torch
from pydantic import PrivateAttr
from langchain.llms.base import LLM
from llama_cpp import Llama
from langchain_openai import ChatOpenAI
from .llm_config import (
    OPENAI_API_KEY
)


# ==== CLASSE PARA TRANSFORMERS ====
class TransformersLLM(LLM):
    _model: any = PrivateAttr()
    _tokenizer: any = PrivateAttr()

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer

    def _call(self, prompt: str, stop=None) -> str:
        # Adaptação para modelo instruído por chat (como o Gemma)
        chat = [{"role": "user", "content": prompt}]
        chat_prompt = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        inputs = self._tokenizer(chat_prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
            result = self._tokenizer.decode(
                output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
        )
        return result[len(chat_prompt):].strip()


    @property
    def _llm_type(self) -> str:
        return "transformers-local"

# ==== GGUF ====
class GGUFLLM(LLM):
    _llm: any = PrivateAttr()

    def __init__(self, model_path: str, n_ctx=2048, **kwargs):
        super().__init__(**kwargs)
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx)

    def _call(self, prompt: str, stop=None) -> str:
        result = self._llm(prompt, max_tokens=300, temperature=0.3, top_p=0.9)
        return result["choices"][0]["text"].strip()

    @property
    def _llm_type(self) -> str:
        return "gguf-local"


# # # ===== Classe para OpenAI ====
# class OpenAILLM(LLM):
#     _llm: any = PrivateAttr()

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._llm = ChatOpenAI(
#             model=OPENAI_MODEL_NAME,
#             api_key=OPENAI_API_KEY,
#             temperature=0,
#             max_tokens=800
#         )

#     def _call(self, prompt: str, stop=None) -> str:
#         """
#         Executa a chamada ao modelo OpenAI.
#         Espera um prompt em string e retorna uma resposta em string.
#         """
#         response = self._llm.invoke(prompt)
#         return response.content if hasattr(response, "content") else response

#     @property
#     def _llm_type(self) -> str:
#         return "openai"

class OpenAILLM(LLM):
    """
    Wrapper do ChatOpenAI que aceita o nome do modelo dinamicamente e
    lida com a continuação automática de respostas longas.
    """
    _llm: any = PrivateAttr()

    def __init__(self, model_name: str, temperature: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if not OPENAI_API_KEY:
            raise ValueError("A chave de API da OpenAI não foi encontrada no ambiente.")
        
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        """
        Executa a chamada ao modelo. 
        A função `ask_with_model` já garante que o 'prompt' aqui é uma lista de mensagens bem formatada.
        """
        full_answer = ""
        messages = prompt # O 'prompt' recebido aqui já é a lista de mensagens.
        
        while True:
            # Passamos a lista de mensagens diretamente para o invoke.
            resp = self._llm.invoke(messages, stop=stop, **kwargs)
            content = resp.content if hasattr(resp, "content") else str(resp)
            full_answer += content

            finish_reason = resp.response_metadata.get("finish_reason") if hasattr(resp, "response_metadata") else None

            # Se a resposta não foi truncada, encerramos o loop.
            if finish_reason != "length":
                break

            # Prepara para a chamada de continuação, adicionando a resposta do assistente ao histórico
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Continue de onde parou, por favor."})
        
        return full_answer


    @property
    def _llm_type(self) -> str:
        return "openai"