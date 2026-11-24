"""Suavização exponencial tripla (Holt-Winters multiplicativo), em Python puro —
sem numpy/scipy/statsmodels. Pensado pra rodar embarcado em agentes leves (sem
dependência pesada de estatística) e pra ser fácil de ler/auditar linha a linha.

Resolve dois problemas de previsão de faturamento mensal num negócio sazonal
(ex.: restaurantes, varejo):

1. `holt_winters_multiplicativo` / `prever_holt_winters`: previsão padrão quando
   já existe histórico suficiente (>= 24 meses, 2 temporadas completas) pra
   estimar nível, tendência E sazonalidade própria do negócio.

2. `calcular_indices_sazonais_emprestados` / `projetar_com_sazonal_emprestado`:
   o problema do "cold start" — uma unidade nova (loja recém-aberta, produto
   novo) não tem 2 anos de histórico pra estimar sazonalidade nenhuma. A
   solução aqui é calcular a sazonalidade típica a partir de outras unidades
   já maduras (cada uma normalizada pra tirar o efeito de escala, só sobra o
   "formato" da curva ao longo do ano) e emprestar esse formato pra projetar
   a unidade nova, enquanto ajusta nível/tendência com o pouco de dado real
   que ela já tem.
"""

from __future__ import annotations


def holt_winters_multiplicativo(
    meses_completos: list[dict], alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.3
) -> dict:
    """meses_completos: lista CRONOLÓGICA e SEM BURACO de mês, cada item
    {"ano": int, "mes": int, "faturamento": float}. Precisa de pelo menos 24
    pontos (2 temporadas) pra inicializar nível/tendência/sazonal de forma
    confiável — com menos que isso, usar `projetar_com_sazonal_emprestado`.

    Os índices sazonais ficam indexados pelo MÊS DO CALENDÁRIO (1-12), não
    pela posição na lista — importante porque o histórico pode começar em
    qualquer mês do ano, não necessariamente em janeiro.
    """
    periodo = 12
    ys = [float(m["faturamento"] or 0) for m in meses_completos]
    meses_cal = [int(m["mes"]) for m in meses_completos]
    n = len(ys)
    if n < 2 * periodo:
        raise ValueError(f"holt_winters_multiplicativo precisa de >= {2 * periodo} meses, recebeu {n}")

    media_temp1 = sum(ys[:periodo]) / periodo
    media_temp2 = sum(ys[periodo:2 * periodo]) / periodo
    nivel = media_temp1
    tendencia = (media_temp2 - media_temp1) / periodo
    sazonais = {meses_cal[i]: (ys[i] / media_temp1 if media_temp1 else 1.0) for i in range(periodo)}

    ajustados: list[float | None] = [None] * n
    for t in range(n):
        mes_cal = meses_cal[t]
        if t < periodo:
            ajustados[t] = (nivel + tendencia * t) * sazonais.get(mes_cal, 1.0)
            continue
        sazonal_anterior = sazonais.get(mes_cal, 1.0)
        nivel_anterior = nivel
        nivel = alpha * (ys[t] / sazonal_anterior) + (1 - alpha) * (nivel + tendencia)
        tendencia = beta * (nivel - nivel_anterior) + (1 - beta) * tendencia
        sazonais[mes_cal] = gamma * (ys[t] / nivel) + (1 - gamma) * sazonal_anterior
        ajustados[t] = (nivel_anterior + tendencia) * sazonal_anterior

    return {"nivel": nivel, "tendencia": tendencia, "sazonais": sazonais, "ajustados": ajustados, "n": n}


def prever_holt_winters(modelo: dict, mes_alvo: int, passos_a_frente: int) -> float:
    """passos_a_frente: quantos meses depois do ÚLTIMO ponto da série usada pra
    treinar (0 = o mês seguinte ao último ponto)."""
    sazonal = modelo["sazonais"].get(mes_alvo, 1.0)
    return max((modelo["nivel"] + modelo["tendencia"] * (passos_a_frente + 1)) * sazonal, 0.0)


def calcular_indices_sazonais_emprestados(series_unidades_maduras: list[list[dict]]) -> dict:
    """series_unidades_maduras: lista de `meses_completos` (uma lista por
    unidade MADURA, ou seja já com >= 24 meses de histórico real — unidades
    com menos são ignoradas aqui). Cada unidade normaliza o próprio índice
    sazonal (média = 1, tira o efeito de escala — só queremos o FORMATO da
    curva, não o faturamento absoluto) antes de entrar na média entre
    unidades. Resultado: 'sazonalidade típica' do negócio, emprestável pra
    uma unidade nova sem histórico suficiente pra estimar a própria.
    """
    acumulado: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for meses_completos in series_unidades_maduras:
        if len(meses_completos) < 24:
            continue
        try:
            modelo = holt_winters_multiplicativo(meses_completos)
        except ValueError:
            continue
        indices = modelo["sazonais"]
        if not indices:
            continue
        media_indices = sum(indices.values()) / len(indices)
        if media_indices <= 0:
            continue
        for mes_cal, idx in indices.items():
            acumulado[mes_cal].append(idx / media_indices)
    return {mes: (sum(vals) / len(vals) if vals else 1.0) for mes, vals in acumulado.items()}


def projetar_com_sazonal_emprestado(
    meses_completos: list[dict], indices_emprestados: dict, pedidos: list[tuple[int, int]]
) -> dict:
    """Pra uma unidade com histórico curto (< 24 meses, inclusive bem curto —
    3 a 8 meses, caso de unidade recém-aberta): deseasonaliza a série própria
    usando o índice EMPRESTADO (calculado de unidades maduras), ajusta
    nível+tendência simples (Holt duplo, sem sazonal próprio — não há dado
    suficiente pra estimar isso de verdade) em cima da série já
    deseasonalizada, e reaplica o índice emprestado na hora de projetar pra
    devolver a sazonalidade "típica" ao resultado final.

    `pedidos`: lista de (mes_alvo_calendario, passos_a_frente) — passos_a_frente=0
    é o mês seguinte ao último ponto da série.
    Retorna dict {(mes_alvo, passos_a_frente): valor_previsto}.
    """
    ys = [float(m["faturamento"] or 0) for m in meses_completos]
    meses_cal = [int(m["mes"]) for m in meses_completos]
    if not ys:
        return {p: 0.0 for p in pedidos}

    desaz = [y / indices_emprestados.get(mc, 1.0) for y, mc in zip(ys, meses_cal)]

    alpha, beta = 0.4, 0.2
    nivel = desaz[0]
    tendencia = (desaz[1] - desaz[0]) if len(desaz) > 1 else 0.0
    for t in range(1, len(desaz)):
        nivel_anterior = nivel
        nivel = alpha * desaz[t] + (1 - alpha) * (nivel + tendencia)
        tendencia = beta * (nivel - nivel_anterior) + (1 - beta) * tendencia

    resultado = {}
    for mes_alvo, passos in pedidos:
        valor_deseaz = nivel + tendencia * (passos + 1)
        resultado[(mes_alvo, passos)] = max(valor_deseaz * indices_emprestados.get(mes_alvo, 1.0), 0.0)
    return resultado
