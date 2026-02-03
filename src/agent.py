# agent.py
import os
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()

# ReActAgent import varia por versão -> fallback
try:
    from llama_index.core.agent.workflow import ReActAgent
except Exception:
    from llama_index.core.agent import ReActAgent

from agent_tools import TOOLS



def get_agent():
    # LlamaIndex usa Settings.llm (padrão)
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
    )
    
    system_prompt = """
Você é um Analista de IA Sênior especializado em vendas.

Siga EXATAMENTE este formato:
Thought: <seu raciocínio curto>
Action: <nome_da_ferramenta ou None>
Action Input: <texto ou string vazia>
Observation: <resultado da ferramenta>
... (repita Thought/Action/Action Input/Observation quando precisar)
Final Answer: <resposta final em português, objetiva. AO FINAL, cite explicitamente quais ferramentas você utilizou.>

Regras:
- Sempre que o usuário perguntar "quais ferramentas foram usadas", liste os nomes das ferramentas que você acionou durante o raciocínio.
- Para perguntas múltiplas, chame as ferramentas necessárias uma após a outra e combine no Final Answer.
- Use 'consulta_geral' apenas se não existir ferramenta específica.
""".strip()

    agent = ReActAgent(
        tools=TOOLS,
        llm=Settings.llm,
        system_prompt=system_prompt,
    )

    return agent
