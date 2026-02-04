from llama_index.core.tools import FunctionTool
from llama_index.experimental.query_engine import PandasQueryEngine
import analytics as t

query_engine = PandasQueryEngine(df=t.df, verbose=False)

def tool_consulta_geral(pergunta: str) -> str:
    """
    Útil para perguntas complexas sobre o dataset que não possuem ferramentas específicas.
    Passe a pergunta completa em português.
    """
    
    resposta = query_engine.query(pergunta)
    print("[Texto gerado apartir de pandasQueries]")
    return str(resposta)

# =========================
# 1) Desempenho de vendas e acurácia de planejamento
# =========================
def tool_calcular_acuracia_planejamento() -> str:
    """
    Calcula o desvio percentual (pct_desvio) entre planned_quantity e actual_quantity
    e retorna colunas-chave para análise.
    """
    df_out = t.calcular_acuracia_planejamento(t.df.copy())
    return df_out.head(20).to_string(index=False)


def tool_identificar_ruptura_ou_excesso(threshold: float = 0.2) -> str:
    """
    Identifica linhas em que actual_quantity diverge muito de planned_quantity.
    threshold=0.2 significa ±20%.
    """
    alertas = t.identificar_ruptura_ou_excesso(t.df.copy(), threshold=threshold)
    if alertas.empty:
        return f"Nenhum alerta encontrado com threshold={threshold:.2f}."
    return alertas.head(50).to_string(index=False)


# =========================
# 2) Impacto de promoções por produto
# =========================
def tool_impacto_promocao_por_produto() -> str:
    """
    Compara médias de volume, preço e nível de serviço por product_id e promotion_type.
    """
    analise = t.impacto_promocao_por_produto(t.df.copy())
    if analise.empty:
        return "Sem dados para analisar impacto de promoção por produto."
    return analise.head(50).to_string(index=False)


# =========================
# 3) Ranking e Curva ABC (Pareto)
# =========================
def tool_ranking_receita_por_local() -> str:
    """
    Ranking de receita real (actual_quantity * actual_price) por local.
    """
    ranking = t.ranking_receita_por_local(t.df.copy())
    return ranking.head(20).to_string()


def tool_produtos_mais_vendidos(top_n: int = 10) -> str:
    """
    Retorna os top N produtos por volume total vendido (actual_quantity).
    """
    top = t.produtos_mais_vendidos(t.df.copy(), top_n=top_n)
    return top.to_string()


# =========================
# 4) Nível de serviço
# =========================
def tool_analisar_degradacao_servico(min_service_level: float = 0.95) -> str:
    """
    Lista transações onde service_level ficou abaixo de um mínimo.
    """
    df_bad = t.analisar_degradacao_servico(t.df.copy(), min_service_level=min_service_level)
    if df_bad.empty:
        return f"Nenhuma transação abaixo de min_service_level={min_service_level:.2f}."
    return df_bad.head(50).to_string(index=False)


# =========================
# 5) Perguntas do README (helpers)
# =========================
def tool_top_entidades(
    group_by_col: str = "product_id",
    metric: str = "actual_quantity",
    top_n: int = 5,
) -> dict:
    """
    Top N entidades (ex: product_id/local) pelo somatório de uma métrica.
    """
    return t.get_top_performing_entities(t.df, group_by_col=group_by_col, metric=metric, top_n=top_n)


def tool_vendas_por_periodo(start_date: str, end_date: str) -> dict:
    """
    Total de vendas (actual_quantity) em um período.
    Datas no formato YYYY-MM-DD.
    """
    return t.get_total_sales_period(t.df.copy(), start_date=start_date, end_date=end_date)


def tool_gap_planejamento() -> dict:
    """
    Diferença entre planejado e realizado (gap_total, mape_medio e tendência).
    """
    return t.analyze_planning_gap(t.df.copy())


# =========================
# 5b) Métricas extras (para evitar "chutes" do PandasQueryEngine)
# =========================
def tool_promocao_share() -> dict:
    """
    Retorna qual % das vendas ocorreu com promoção.
    (linhas, volume e receita)
    """
    return t.get_promocao_share(t.df.copy())


def tool_preco_medio_geral() -> dict:
    """
    Retorna o preço médio geral (actual_price).
    """
    return t.get_preco_medio_geral(t.df.copy())


def tool_produto_maior_receita() -> dict:
    """
    Retorna o produto com maior receita total.
    Receita = soma(actual_quantity * actual_price) por produto.
    """
    return t.get_produto_maior_receita(t.df.copy())

# =========================
# 6) Elasticidade / Promoção (resumo por promotion_type)
# =========================
def tool_impacto_promocao() -> dict:
    """
    Compara médias com e sem promoção por promotion_type.
    """
    return t.analyze_promotion_impact(t.df.copy())


# =========================
# 7) Saúde Logística
# =========================
def tool_risco_servico(threshold: float = 0.85) -> dict:
    """
    Identifica combinações local+produto com nível de serviço médio crítico.
    """
    return t.check_service_risk(t.df.copy(), threshold=threshold)

# =========================
# 8) Relatório executivo (texto + PDF)
# =========================
def tool_gerar_relatorio(top_n: int = 5) -> str:
    """
    Gera um relatório executivo em texto com os principais indicadores do dataset.
    """
    return t.gerar_relatorio_executivo(t.df.copy(), top_n=top_n)


def tool_gerar_relatorio_pdf(top_n: int = 5, output_path: str = "reports/relatorio_executivo.pdf") -> str:
    """
    Gera o relatório executivo e salva em PDF.
    Retorna o caminho do arquivo gerado.
    """
    return t.gerar_relatorio_pdf(t.df.copy(), output_path=output_path, top_n=top_n)


def tool_q1_produto_maior_desvio_absoluto() -> dict:
    return t.q1_produto_maior_desvio_absoluto(t.df.copy())

def tool_q2_local_maior_desvio_percentual_medio() -> dict:
    return t.q2_local_maior_desvio_percentual_medio(t.df.copy())

def tool_q3_top5_volume_maior_preco_medio() -> dict:
    return t.q3_top5_volume_maior_preco_medio(t.df.copy())

def tool_q4_mes_menor_volume() -> dict:
    return t.q4_mes_menor_volume(t.df.copy())

def tool_q5_top10_volume_menor_receita_unitaria() -> dict:
    return t.q5_top10_volume_menor_receita_unitaria(t.df.copy())

def tool_q6_media_volume_diario() -> dict:
    return t.q6_media_volume_diario(t.df.copy())

def tool_q7_maior_delta_volume_com_promocao() -> dict:
    return t.q7_maior_delta_volume_com_promocao(t.df.copy())

def tool_q8_share_receita_por_local() -> dict:
    return t.q8_share_receita_por_local(t.df.copy())

def tool_q9_maior_pico_diario_produto() -> dict:
    return t.q9_maior_pico_diario_produto(t.df.copy())

def tool_q10_impacto_remover_top_receita() -> dict:
    return t.q10_impacto_remover_top_receita(t.df.copy())




TOOLS = [
    FunctionTool.from_defaults(fn=tool_consulta_geral, name="consulta_geral"),
        # 1) Planejamento / ruptura
    FunctionTool.from_defaults(fn=tool_calcular_acuracia_planejamento, name="calcular_acuracia_planejamento"),
    FunctionTool.from_defaults(fn=tool_identificar_ruptura_ou_excesso, name="identificar_ruptura_ou_excesso"),

    # 2) Promoção por produto
    FunctionTool.from_defaults(fn=tool_impacto_promocao_por_produto, name="impacto_promocao_por_produto"),

    # 3) Ranking / Top produtos
    FunctionTool.from_defaults(fn=tool_ranking_receita_por_local, name="ranking_receita_por_local"),
    FunctionTool.from_defaults(fn=tool_produtos_mais_vendidos, name="produtos_mais_vendidos"),

    # 4) Serviço
    FunctionTool.from_defaults(fn=tool_analisar_degradacao_servico, name="analisar_degradacao_servico"),

    # 5) Readme helpers
    FunctionTool.from_defaults(fn=tool_top_entidades, name="top_entidades"),
    FunctionTool.from_defaults(fn=tool_vendas_por_periodo, name="vendas_por_periodo"),
    FunctionTool.from_defaults(fn=tool_gap_planejamento, name="gap_planejamento"),

     # 5b) extras
    FunctionTool.from_defaults(fn=tool_promocao_share, name="promocao_share"),
    FunctionTool.from_defaults(fn=tool_preco_medio_geral, name="preco_medio_geral"),
    FunctionTool.from_defaults(fn=tool_produto_maior_receita, name="produto_maior_receita"),

    # 6) Promoção (por tipo)
    FunctionTool.from_defaults(fn=tool_impacto_promocao, name="impacto_promocao"),

    # 7) Risco serviço
    FunctionTool.from_defaults(fn=tool_risco_servico, name="risco_servico"),
    # 8) relatorio
    FunctionTool.from_defaults(fn=tool_gerar_relatorio, name="gerar_relatorio"),
    FunctionTool.from_defaults(fn=tool_gerar_relatorio_pdf, name="gerar_relatorio_pdf"),

    FunctionTool.from_defaults(fn=tool_q1_produto_maior_desvio_absoluto, name="produto_maior_desvio_absoluto"),
    FunctionTool.from_defaults(fn=tool_q2_local_maior_desvio_percentual_medio, name="local_maior_desvio_percentual_medio"),
    FunctionTool.from_defaults(fn=tool_q3_top5_volume_maior_preco_medio, name="top5_volume_maior_preco_medio"),
    FunctionTool.from_defaults(fn=tool_q4_mes_menor_volume, name="mes_menor_volume"),
    FunctionTool.from_defaults(fn=tool_q5_top10_volume_menor_receita_unitaria, name="top10_volume_menor_receita_unitaria"),
    FunctionTool.from_defaults(fn=tool_q6_media_volume_diario, name="media_volume_diario"),
    FunctionTool.from_defaults(fn=tool_q7_maior_delta_volume_com_promocao, name="maior_delta_promocao"),
    FunctionTool.from_defaults(fn=tool_q8_share_receita_por_local, name="share_receita_por_local"),
    FunctionTool.from_defaults(fn=tool_q9_maior_pico_diario_produto, name="pico_diario_produto"),
    FunctionTool.from_defaults(fn=tool_q10_impacto_remover_top_receita, name="impacto_remover_top_receita"),
]
