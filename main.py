import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool 

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# 1. Configuração do Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
)

# 2. Ferramenta de Busca
@tool("Ferramenta de Busca Web")
def web_search(search_query: str) -> str:
    """Utilize esta ferramenta para buscar informações sobre empresas, construtoras e obras na internet e redes sociais."""
    search = DuckDuckGoSearchRun()
    return search.run(search_query)

# 3. AGENTE PESQUISADOR (SDR Especializado em Construção Civil)
mapeador_obras = Agent(
    role='Analista de Inteligência de Mercado e Prospecção B2B',
    goal='Localizar potenciais compradores, construtoras locais e projetos de engenharia que demandam telhas termoacústicas',
    backstory="""Você é um especialista em mapear o mercado da construção civil, com foco em identificar empresas que realizam obras de galpões, coberturas residenciais, comerciais ou industriais.
    Sua prioridade absoluta é encontrar construtoras locais, empreiteiras de médio porte, depósitos de materiais de construção parceiros e escritórios de arquitetura/engenharia com presença ativa no Instagram ou Google Maps.
    Você sabe identificar sinais de que uma empresa realiza obras que utilizam telhas termoacústicas (como construções com estruturas metálicas, galpões logísticos ou reformas térmicas).
    
    FILTROS RÍGIDOS E PROIBIÇÕES:
    - Priorize estritamente construtoras locais, regionais, empreiteiras e distribuidoras regionais.
    - É estritamente PROIBIDO listar gigantes do setor ou multinacionais (ex: MRV, Cyrela, Tenda, Camargo Corrêa) que negociam diretamente em escala industrial.
    - Foque em negócios que utilizam fornecedores regionais para suas obras.""",
    tools=[web_search], 
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 4. TAREFA DE MAPEAMENTO
task_mapear_clientes = Task(
    description="""Pesquise 10 potenciais clientes ou parceiros comerciais do nicho de {nicho} localizados em/no {localizacao}.
    FOCO: Encontrar perfis no Instagram, sites ou Google Maps de empresas que mostram obras em andamento, projetos de galpões, coberturas ou reformas recentes.
    
    CRITÉRIOS DE SELEÇÃO:
    1. Construtoras, galpões comerciais em desenvolvimento, empreiteiras locais ou revendas de ferro/aço/telhas.
    2. Sinais de fit com o produto: postagens de estruturas metálicas, projetos de galpões, telhados industriais ou obras residenciais modernas.
    
    EXCLUA: Grandes incorporadoras de nível nacional ou indústrias petroquímicas.
    
    Para cada lead encontrado, identifique: Nome da Empresa/Projeto, Cidade/Estado, Link (Instagram ou Site) e o 'Sinal de Fit' (ex: 'Postou foto de obra com estrutura metálica pronta para receber cobertura' ou 'Construtora focada em galpões comerciais').""",
    agent=mapeador_obras,
    expected_output="Uma lista estruturada contendo: Nome da Empresa/Lead, Localização, Link de Contato, Sinal de Fit/Oportunidade e Abordagem Recomendada para venda de telhas termoacústicas.",
)

# 5. A EQUIPE
equipe_vendas_telhas = Crew(
    agents=[mapeador_obras],           
    tasks=[task_mapear_clientes],         
    process=Process.sequential,
    verbose=True
)

# 6. EXECUÇÃO
if __name__ == "__main__":
    # Exemplo focado no Norte do país, mas pode ser alterado conforme sua estratégia
    inputs = {
        'nicho': 'Construtoras de Médio Porte, Empreiteiras de Estruturas Metálicas e Galpões Comerciais',
        'localizacao': 'Pará'
    }

    print(f"\n### 🚀 Iniciando Prospecção de Clientes para Telhas Termoacústicas no nicho de: {inputs['nicho']} ###\n")
    
    try:
        resultado = equipe_vendas_telhas.kickoff(inputs=inputs)
        print("\n\n########################")
        print("## LEADS ENCONTRADOS PARA TELHAS TERMOACÚSTICAS ##")
        print(resultado)
    except Exception as e:
        print(f"\n❌ Ocorreu um erro na execução: {e}")
