# Nexus: Um Agente Pessoal de Monitoramento Contextual de Atividades de Desenvolvimento com IA

#### Aluno: [Paulo Moura](https://github.com/paulofjm)

#### Orientadora: [Evelyn Batista](https://github.com/evysb)

---

Trabalho apresentado ao curso [BI MASTER](https://ica.ele.puc-rio.br/cursos/mba-bi-master/) como pré‑requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

- [Link para o código](https://github.com/paulofjm/Nexus_project)

---

### Resumo

O **Nexus** é um sistema multiagente que monitora continuamente a estação de trabalho do desenvolvedor, coletando evidências de contexto como janelas ativas, capturas de tela, texto extraído via OCR, estatísticas de hardware e versões do código‑fonte para construir uma memória hierárquica consultável em tempo real. Utilizando embeddings *Sentence‑Transformers* e um índice vetorial FAISS organizado em camadas de curto, médio e longo prazo, o Nexus responde a perguntas em linguagem natural, sugere correções de código, executa ferramentas locais e registra insights do fluxo de trabalho. A orquestração é feita com **LangGraph**, permitindo roteamento dinâmico entre múltiplos LLMs (locais ou OpenAI) e ferramentas personalizadas. O resultado é um assistente que reduz perda de contexto, acelera troubleshooting e cria documentação viva do projeto.

### Abstract *(English)*

**Nexus** is a multi‑agent system that continuously monitors a developer’s workstation, capturing contextual evidence active windows, screenshots, OCR‑extracted text, hardware telemetry and live code snapshots to build a hierarchical, queryable memory in real time. Leveraging Sentence‑Transformers embeddings and a layered FAISS vector index, Nexus answers natural‑language questions, suggests code fixes, triggers local tools and logs workflow insights. **LangGraph** orchestrates dynamic routing among multiple LLMs (local or OpenAI) and custom tools. The result is an assistant that minimizes context loss, speeds up troubleshooting and generates living documentation of the project.

### 1. Introdução

Projetos de software modernos envolvem múltiplas janelas, IDEs, navegadores e documentação dispersa. Desenvolvedores perdem tempo alternando contextos e recuperando informações que já produziram. Paralelamente, LLMs despontaram como poderosos assistentes, mas carecem de contexto de execução local. O Nexus nasce para preencher essa lacuna: capturar a "memória operacional" do desenvolvedor e disponibilizá‑la a um agente conversacional capaz de responder perguntas contextuais, executar ações e evoluir com o uso.

### 2. Modelagem

- **Arquitetura de Módulos**: `core` (contexto e armazenamento), `models` (LLMs e embeddings), `nodes_graph` (LangGraph), `utils` (OCR, monitoramento, logs).
- **Monitoramento Contínuo**: *thread* que a cada 5 s verifica atividade relevante, captura telas, roda OCR (Tesseract/Donut/TroCR) e coleta métricas de CPU/RAM.
- **Versionamento de Código**: *watchdog* detecta alterações em arquivos `.py`, gera snapshot concatenado, calcula hash SHA‑256 e grava em `logs/code_versions/`.
- **Memória Hierárquica**: 20 registros → 1 bloco curto; 36 curtos → 1 médio; 50 médios → 1 longo; todos vetorizados no FAISS.
- **LangGraph RAG**: grafo de estados com nós para interpretação de intenção, consulta à memória, RAG de código, execução de ferramentas e fallback. O roteador escolhe entre modelos “Router”, “Activity” e “Code”.
- **Ferramentas Customizadas**: captura de screenshot, listagem de arquivos, abertura de programas, expansível via `nexus_tools_retriever`.

### 3. Resultados

# Análise Quantitativa de Desempenho de Modelos no Agente Nexus

## 1. Metodologia de Análise

A análise a seguir foi conduzida correlacionando dados de múltiplos arquivos de log gerados pelo sistema Nexus em 2025-07-08. As principais fontes de dados foram:

- **monitor_log.json**: Para rastrear a sequência de execução dos nós do grafo, os modelos de linguagem (LLMs) utilizados em cada etapa e o tempo de inferência individual de cada chamada.
- **turns.json**: Para obter a pergunta exata do usuário e a resposta final gerada pelo agente, incluindo o tempo total de processamento reportado.
- **activity_raw_buffer.json**: Para extrair dados de utilização de recursos do sistema (CPU e memória) durante os períodos de teste.

A análise foca nos casos de uso mais recentes para avaliar o desempenho dos diferentes LLMs configurados para os papéis de "Agente Roteador" e "Agente de Geração". A utilização de GPU não foi registrada nos logs, portanto não está incluída nesta análise.

## 2. Análise de Casos de Uso (Recentes)

Foram analisadas três interações distintas que ilustram o comportamento do sistema com diferentes configurações de modelo.

### Caso de Estudo 1: Pergunta sobre Código com Roteador Ineficaz

- **Pergunta do Usuário:** "explique cada nó do langgraph_nodes.py"
- **Timestamp da Interação:** ~20:09

**Etapa 1: Roteamento (Classificação de Intenção)**
- Modelo Utilizado: `gemma:2b (Ollama)`
- Tempo de Inferência: 5.89 segundos
- Resultado da Classificação: `temporal`
- **Análise:** Falha Crítica. O modelo falhou em identificar a intenção `codigo` e classificou incorretamente como `temporal`, uma busca por atividades em um período de tempo.


**Etapa 2: Geração de Resposta (com Contexto Errado)**
- Modelo Utilizado: `llama3.1:8b (Ollama)`
- Tempo de Inferência: 12.67 segundos
- Contexto Recebido: Devido ao erro de roteamento, o modelo recebeu um resumo das atividades recentes do usuário em vez do resumo do código do projeto.
- Resposta Final: Genérica e incorreta. Descreveu o que são "nós" de forma abstrata, sem qualquer relação com o arquivo langgraph_nodes.py.
- Tempo Total Reportado: 166.62 segundos (valor parece incluir tempo de espera ou atividades em background).

Esse caso evidencia como um erro de roteamento de intenção compromete toda a cadeia do agente. Mesmo que o modelo de geração seja adequado, a resposta sai incorreta por receber contexto irrelevante. Mostra a importância de calibrar bem o roteador, pois ele atua como o elo crítico do fluxo.

---

### Caso de Estudo 2: Pergunta sobre Código com Roteador Eficaz (Modelo de Código Fraco)

- **Pergunta do Usuário:** "explique o main.py"
- **Timestamp da Interação:** ~19:59 (UTC-3)

**Etapa 1: Roteamento (Classificação de Intenção)**
- Modelo Utilizado: `gpt-4o-mini (OpenAI)`
- Tempo de Inferência: 3.15 segundos
- Resultado da Classificação: `codigo`
- **Análise:** Sucesso. O modelo classificou a intenção corretamente, direcionando a pergunta para o fluxo de análise de código.

**Etapa 2: Geração de Resposta (com Contexto Correto)**
- Modelo Utilizado: `deepseek-coder:6.7b-instruct (Ollama)`
- Tempo de Inferência: 102.86 segundos
- Contexto Recebido: O modelo recebeu o resumo do código do projeto, como esperado.
- Resposta Final: Evasiva e de baixa qualidade. (Ex: "...não há muitas informações suficientes para entender...").
- Tempo Total Reportado: 110.27 segundos.

O roteamento funcionou corretamente, mas a fraqueza do modelo de geração de código reduziu a qualidade da resposta. Demonstra que um bom roteador precisa ser acompanhado por modelos especializados para garantir profundidade técnica e clareza explicativa.


---

### Caso de Estudo 3: Pergunta sobre Código com Roteador e Gerador Eficazes (Sucesso Pleno)

- **Pergunta do Usuário:** "o que o código do projeto nexus faz?"
- **Timestamp da Interação:** ~19:58 (UTC-3)

**Etapa 1: Roteamento (Classificação de Intenção)**
- Modelo Utilizado: `gpt-4o-mini (OpenAI)`
- Tempo de Inferência: 1.07 segundos
- Resultado da Classificação: `codigo`
- **Análise:** Sucesso. Roteamento perfeito.

**Etapa 2: Geração de Resposta (com Contexto Correto)**
- Modelo Utilizado: `gpt-4o-mini (OpenAI)`
- Tempo de Inferência: 13.53 segundos
- Contexto Recebido: Resumo do código do projeto.
- Resposta Final: Excelente. Detalhada, precisa e bem estruturada, explicando corretamente todos os componentes do projeto.
- Tempo Total Reportado: 14.6 segundos (soma das inferências do monitor_log.json).

Esse é o cenário ideal: roteamento eficaz e modelo de geração compatível com a tarefa. O resultado foi resposta precisa, estruturada e contextualizada, confirmando que o fluxo só é robusto quando há sinergia entre roteador e gerador, além de baixo tempo de latência.


---

## 3. Tabela Comparativa de Desempenho

| Caso de Estudo | Pergunta                | Modelo Roteador | Tempo Roteador (s) | Roteamento Correto? | Modelo Gerador    | Tempo Gerador (s) | Qualidade da Resposta    |
|----------------|-------------------------|-----------------|--------------------|---------------------|-------------------|-------------------|--------------------------|
| 1              | explique cada nó...     | gemma:2b        | 5.89               | Não                 | llama3.1:8b       | 12.67             | Inútil / Genérica        |
| 2              | explique o main.py      | gpt-4o-mini     | 3.15               | Sim                 | deepseek-coder    | 102.86            | Fraca / Evasiva          |
| 3              | o que o projeto faz?    | gpt-4o-mini     | 1.07               | Sim                 | gpt-4o-mini       | 13.53             | Excelente / Precisa      |

---

## 4. Análise de Utilização de Recursos

Os dados do arquivo `activity_raw_buffer.json` indicam o seguinte sobre o uso de recursos do sistema durante os testes:

- **CPU:** A utilização variou significativamente, com picos de 20.6% e 39.9% durante a execução de tarefas e monitoramento. O uso médio em períodos de menor atividade ficou entre 1.9% e 2.7%.
- **Memória:** O uso de memória do sistema manteve-se consistentemente alto, variando entre 48.0% e 63.3%. Isso sugere que os modelos de linguagem locais, mesmo quando não estão em uso ativo, consomem uma quantidade substancial de RAM.
- **GPU:** Não foram encontrados dados de utilização de GPU nos logs fornecidos.

---

## 5. Conclusões e Recomendações para o TCC

- **Criticidade do Agente Roteador:** A análise demonstra empiricamente que a eficácia do agente especialista depende criticamente da precisão do Agente Roteador. O modelo `gemma:2b` provou ser inadequado para a tarefa de classificação de intenção com o prompt atual, resultando em falhas em cascata. O `gpt-4o-mini`, por outro lado, apresentou 100% de acerto nos casos analisados.

- **Desempenho vs. Custo (Inferência Local vs. API):** Os modelos locais (Ollama) apresentam a vantagem de não terem custo por chamada, mas o `deepseek-coder` demonstrou um tempo de inferência significativamente alto (102.86s) e uma qualidade de resposta inferior para uma tarefa complexa. O `gpt-4o-mini` ofereceu um equilíbrio superior entre tempo de resposta (1-3s para roteamento, 13s para geração) e qualidade excepcional.

- **Impacto do Contexto:** O Caso de Estudo 1 prova que mesmo um modelo potente como o `llama3.1:8b` é ineficaz se alimentado com um contexto incorreto devido a uma falha de roteamento. A arquitetura de RAG (Recuperação Aumentada por Geração) só funciona se a etapa de roteamento inicial for precisa.

- **Recomendação Final:** Para a implementação final do projeto, recomenda-se fixar o `gpt-4o-mini` como o "Agente Roteador" padrão. Essa única mudança garantirá a estabilidade do sistema e permitirá uma avaliação mais justa do desempenho dos diferentes modelos de geração (locais ou via API) para as tarefas de Atividade e Código, pois eles sempre receberão o direcionamento e o contexto corretos.

---

### 6. Demonstração da Interface do Nexus Agent

Abaixo estão algumas capturas de tela da interface desenvolvida, ilustrando como o usuário interage com o sistema:

### Tela – Configuração de Modelos
![Configuração de Modelos](imgs/nexus_ft1.png)

### Tela – Chat de Atividades
![Chat de Atividades](imgs/nexus_ft2.png)

### Tela – Chat de Código
![Chat de Código](imgs/nexus_ft3.png)









### 4. Conclusões

O Nexus comprova que a combinação de monitoramento local, memória vetorial hierárquica e orquestração de LLMs fornece ganho real de produtividade ao desenvolvedor, reduzindo a busca manual por contexto e acelerando a decisão. Futuras evoluções incluem: tradução de voz em tempo real, integração com IDEs, armazenamento criptografado e ajuste fino de modelos locais voltados a código.

---

Matrícula: 231.100.980

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós‑Graduação *Business Intelligence Master*

\===============================================================================

## Requisitos

- **Python 3.9 ou superior**
- **Tesseract OCR** instalado e no `PATH`
- GPU NVIDIA opcional para aceleração (CUDA 11.8+)
- **Ollama** instalado e no `PATH` para modeslos locais (https://ollama.com/)


### Instalação do Tesseract

**Windows**

1. Baixe o instalador em: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Durante a instalação marque **“Add to PATH”**.

**Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Instalação do projeto

```bash
git clone https://github.com/paulofjm/Nexus_project.git
cd Nexus_project
python -m venv env_nexus
source env_nexus/bin/activate # Windows: env_nexus\Scripts\activate
pip install -r requirements.txt
```

Crie um arquivo `.env` e defina pelo menos:

```env
LLM_MODE=local      # ou openai
OPENAI_API_KEY=sk-...
```

### Execução

```bash
python main.py
```

O script carregará os modelos configurados, iniciará o monitoramento em segundo plano e abrirá um prompt interativo.

Para redefinir a memória e logs use:

```bash
python reset_agent.py
```

---

> **Observação**: a utilização de modelos OpenAI pode gerar custos na sua conta. Certifique‑se de revisar as chaves e limites antes de executar em produção.

