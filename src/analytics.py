# tools.py
import pandas as pd

# Carregando o dataset
df = pd.read_csv("data/sales.csv", sep=";", low_memory=False)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
# função para formatação dos numeros
def formatar_grandeza(valor):
    if valor >= 1_000_000_000:
        return f"{valor / 1_000_000_000:.2f} Bilhões"
    elif valor >= 1_000_000:
        return f"{valor / 1_000_000:.2f} Milhões"
    elif valor >= 1_000:
        return f"{valor / 1_000:.2f} Mil"
    return str(valor)

# 1 - analise de desempenho de vendas e acuracia de planejamento
def calcular_acuracia_planejamento(df):
    """
    Calcula a diferença percentual entre o planejado e o realizado.
    Útil para identificar erros de previsão de demanda.
    """
    df['variacao_quantidade'] = df['actual_quantity'] - df['planned_quantity']
    df['pct_desvio'] = (df['variacao_quantidade'] / df['planned_quantity']) * 100
    return df[['product_id', 'date', 'planned_quantity', 'actual_quantity', 'pct_desvio']]

def identificar_ruptura_ou_excesso(df, threshold=0.2):
    """
    Identifica casos onde a venda real foi muito abaixo (risco de excesso)
    ou muito acima (risco de ruptura/falta de estoque) do planejado.
    """
    df['razao_real_plan'] = df['actual_quantity'] / df['planned_quantity']
    alertas = df[(df['razao_real_plan'] < (1 - threshold)) | (df['razao_real_plan'] > (1 + threshold))]
    return alertas

#2 - Análise de Impacto de Promoções
def impacto_promocao_por_produto(df):
    """
    Compara o volume médio de vendas e o preço médio em dias com promoção vs dias sem.
    Responde: "A promoção aumentou o volume o suficiente para justificar o preço menor?"
    """
    analise = df.groupby(['product_id', 'promotion_type']).agg({
        'actual_quantity': 'mean',
        'actual_price': 'mean',
        'service_level': 'mean'
    }).reset_index()
    return analise

# 3 - Ranking e Curva ABC (Pareto)
def ranking_receita_por_local(df):
    """
    Calcula a receita real (quantidade * preço real) agrupada por local.
    """
    df['receita_real'] = df['actual_quantity'] * df['actual_price']
    ranking = df.groupby('local')['receita_real'].sum().sort_values(ascending=False)
    return ranking

def produtos_mais_vendidos(df, top_n=10):
    """
    Retorna os N produtos com maior volume de vendas real.
    """
    return df.groupby('product_id')['actual_quantity'].sum().nlargest(top_n)

# 4 - Análise de Nível de Serviço
def analisar_degradacao_servico(df, min_service_level=0.95):
    """
    Filtra transações onde o nível de serviço ficou abaixo da meta.
    Útil para correlacionar se promoções agressivas pioram o nível de serviço.
    """
    return df[df['service_level'] < min_service_level]

# 5 - perguntas do readme
def get_top_performing_entities(df, group_by_col='product_id', metric='actual_quantity', top_n=5):
    """
    Responde: 'Qual produto foi mais vendido?' ou 'Qual local teve maior volume?'
    """
    return df.groupby(group_by_col)[metric].sum().nlargest(top_n).to_dict()

def get_total_sales_period(df, start_date, end_date):
    """
    Responde: 'Qual foi o total de vendas em determinado período?'
    """
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    total = df.loc[mask, 'actual_quantity'].sum()
    return {"periodo": f"{start_date} a {end_date}", "total_vendas": total}

def analyze_planning_gap(df):
    """
    Responde: 'Qual a diferença entre quantidade planejada e realizada?'
    Calcula o Bias (viés) e o MAPE (erro percentual) para o analista.
    """
    df['gap'] = df['actual_quantity'] - df['planned_quantity']
    df['abs_gap_pct'] = (abs(df['gap']) / df['planned_quantity']).replace([float('inf'), -float('inf')], 0)
    
    stats = {
        "gap_total": df['gap'].sum(),
        "mape_medio": f"{df['abs_gap_pct'].mean() * 100:.2f}%",
        "tendencia": "Subestimado" if df['gap'].sum() > 0 else "Superestimado"
    }
    return stats

# 6 - Elasticidade e Promoção
def analyze_promotion_impact(df):
    """
    Compara performance com e sem promoção.
    """
    report = df.groupby('promotion_type').agg({
        'actual_quantity': 'mean',
        'actual_price': 'mean',
        'service_level': 'mean'
    }).rename(columns={
        'actual_quantity': 'media_volume',
        'actual_price': 'preco_medio',
        'service_level': 'nivel_servico_medio'
    })
    return report.to_dict(orient='index')

# 7 - Saúde Logística
def check_service_risk(df, threshold=0.85):
    """
    Identifica locais ou produtos onde o nível de serviço está crítico.
    """
    criticos = df[df['service_level'] < threshold]
    return criticos.groupby(['local', 'product_id'])['service_level'].mean().sort_values().to_dict()
