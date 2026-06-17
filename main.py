import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# ─── 0. Carrega variáveis do .env ────────────────────────────────────────────
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY não encontrada. "
        "Certifique-se de que o arquivo .env está na raiz do projeto."
    )

# ─── 1. Configura o LLM via CrewAI nativo (LiteLLM por baixo) ───────────────
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=gemini_api_key,
    temperature=0.3,
)

# ─── 2. Ferramenta de Busca Web ──────────────────────────────────────────────
@tool("Ferramenta de Busca Web")
def web_search(search_query: str) -> str:
    """
    Busca informações sobre empresas, construtoras, obras, galpões, academias e pessoas 
    pesquisando preços de materiais de construção na internet. Use termos amplos como 
    'preço de telha sanduiche Para', 'construindo galpao PA', 'reforma de telhado', etc.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(search_query, max_results=5):
            results.append(f"- {r['title']}: {r['body']}")
    return "\n".join(results) if results else "Nenhum resultado encontrado."

# ─── 3. Agente Pesquisador (SDR - Construção Civil) ──────────────────────────
mapeador_obras = Agent(
    role="Analista de Inteligência de Mercado e Prospecção B2B e B2C",
    goal=(
        "Localizar potenciais compradores, sejam empresas (PJ) ou consumidores finais (PF), "
        "além de projetos ativos (galpões, academias) que demandam telhas termoacústicas no Pará"
    ),
    backstory="""Você é um especialista em mapear o mercado da construção civil e comportamento de consumo,
com foco em identificar quem necessita de soluções de cobertura térmica. Seu escopo vai além de empresas formais.

Sua prioridade absoluta é encontrar indícios de obras ativas e intenção de compra. Isso inclui:
1. Pessoas Jurídicas (PJ): Construtoras locais, empreiteiras de médio porte, distribuidoras de materiais, e donos de projetos comerciais como novos galpões logísticos, academias, estruturas industriais ou comerciais.
2. Pessoas Físicas (PF): Consumidores finais que estão construindo ou reformando residências, buscando orçamentos de coberturas ou ativamente pesquisando preços de materiais (como "telha sanduíche" ou "telha com isopor") em redes sociais, fóruns e sites locais.

Você sabe monitorar sinais de interesse tanto em sites institucionais quanto em postagens e menções no Instagram, Google Maps e portais regionais.

FILTROS RÍGIDOS:
- Foque estritamente no mercado e público da região do Pará.
- PROIBIDO listar gigantes do setor (MRV, Cyrela, Tenda, Camargo Corrêa).
- Busque tanto por termos técnicos quanto populares (telha sanduíche, telha termoacústica, telhado térmico).""",
    tools=[web_search],
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# ─── 4. Tarefa de Prospecção ─────────────────────────────────────────────────
tarefa_prospeccao = Task(
    description=(
        "Pesquise por potenciais compradores de telhas termoacústicas (telhas sanduíche) na região do Pará. "
        "A busca deve englobar quem está pesquisando preços, pessoas físicas com obras residenciais em andamento, "
        "além de projetos comerciais e industriais de médio/pequeno porte como galpões, academias e construtoras locais. "
        "Para cada lead ou oportunidade encontrada, informe detalhadamente: nome da empresa ou do projeto/pessoa (identificando se é PJ ou PF), "
        "segmento/tipo de obra, indícios de uso ou intenção de compra de telhas termoacústicas e onde a oportunidade foi encontrada "
        "(Google Maps, Instagram, sites locais, fóruns, etc.)."
    ),
    expected_output=(
        "Uma lista de 10 a 15 leads qualificados (PJ e PF) com nome/projeto, segmento (ex: Academia, Galpão, Residencial, Construtora), "
        "localização, canal de origem e a justificativa clara baseada nos indícios de por que são potenciais compradores. "
        "Excluir multinacionais e grandes redes."
    ),
    agent=mapeador_obras,
)

# ─── 5. Monta e executa o Crew ───────────────────────────────────────────────
crew = Crew(
    agents=[mapeador_obras],
    tasks=[tarefa_prospeccao],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    print("\n🚀 Iniciando rastreamento de leads — Telhas Termoacústicas (Pará)\n")
    resultado = crew.kickoff()
    print("\n✅ Leads encontrados:\n")
    print(resultado)