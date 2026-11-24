"""Demo end-to-end com dados SINTÉTICOS (nenhum dado real de cliente) mostrando:

1. Backtest do Holt-Winters multiplicativo: treina com 30 meses de uma unidade
   madura (3 lojas simuladas, cada uma com sazonalidade de restaurante —
   pico em dez/jan, vale em fev/jun), prevê os últimos 6 meses retidos e
   compara contra o valor real (MAPE).

2. Problema do "cold start": uma unidade nova, com só 6 meses de histórico
   (pouco pra estimar sazonalidade própria), recebe a sazonalidade
   "emprestada" das 3 unidades maduras e projeta os próximos 3 meses.

Gera `demo_grafico.png` com a comparação visual.

Rodar: `python demo.py`
"""

import math
import random

from previsao import (
    holt_winters_multiplicativo,
    prever_holt_winters,
    calcular_indices_sazonais_emprestados,
    projetar_com_sazonal_emprestado,
)

random.seed(42)

MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

# Índice sazonal "de restaurante": alta em dez/jan (festas/verão), baixa em fev/jun.
SAZONAL_BASE = {
    1: 1.25, 2: 0.80, 3: 0.95, 4: 0.95, 5: 0.90, 6: 0.80,
    7: 0.95, 8: 1.00, 9: 1.00, 10: 1.05, 11: 1.10, 12: 1.35,
}


def gerar_serie(mes_inicial: int, n_meses: int, nivel_base: float, tendencia_mensal: float, ruido_pct: float) -> list[dict]:
    """Gera uma série mensal sintética com sazonalidade de restaurante + tendência + ruído."""
    serie = []
    mes, ano = mes_inicial, 2023
    for t in range(n_meses):
        sazonal = SAZONAL_BASE[mes] * random.uniform(1 - ruido_pct * 0.3, 1 + ruido_pct * 0.3)
        nivel = nivel_base + tendencia_mensal * t
        ruido = random.uniform(1 - ruido_pct, 1 + ruido_pct)
        faturamento = max(nivel * sazonal * ruido, 0.0)
        serie.append({"ano": ano, "mes": mes, "faturamento": round(faturamento, 2)})
        mes += 1
        if mes > 12:
            mes, ano = 1, ano + 1
    return serie


def mape(reais: list[float], previstos: list[float]) -> float:
    erros = [abs(r - p) / r for r, p in zip(reais, previstos) if r]
    return 100 * sum(erros) / len(erros)


def main():
    # 3 unidades "maduras" (36 meses cada), escalas e ruídos diferentes.
    loja_a = gerar_serie(mes_inicial=1, n_meses=36, nivel_base=80_000, tendencia_mensal=300, ruido_pct=0.06)
    loja_b = gerar_serie(mes_inicial=4, n_meses=36, nivel_base=50_000, tendencia_mensal=150, ruido_pct=0.08)
    loja_c = gerar_serie(mes_inicial=9, n_meses=36, nivel_base=110_000, tendencia_mensal=-100, ruido_pct=0.05)

    # ── 1) Backtest: treina na loja A com 30 meses, prevê os 6 retidos ──
    treino, retido = loja_a[:30], loja_a[30:]
    modelo = holt_winters_multiplicativo(treino)

    previstos = [
        prever_holt_winters(modelo, mes_alvo=m["mes"], passos_a_frente=i)
        for i, m in enumerate(retido)
    ]
    reais = [m["faturamento"] for m in retido]
    erro = mape(reais, previstos)

    print("=== Backtest (loja madura, 30 meses de treino -> 6 meses previstos) ===")
    for m, real, prev in zip(retido, reais, previstos):
        rotulo = f"{MESES_PT[m['mes'] - 1]}/{m['ano']}"
        print(f"  {rotulo}: real R$ {real:>10,.2f}  |  previsto R$ {prev:>10,.2f}")
    print(f"  MAPE no período retido: {erro:.1f}%\n")

    # ── 2) Cold start: unidade nova com só 6 meses, sazonalidade emprestada ──
    loja_nova = gerar_serie(mes_inicial=3, n_meses=6, nivel_base=35_000, tendencia_mensal=400, ruido_pct=0.07)

    indices_emprestados = calcular_indices_sazonais_emprestados([loja_a, loja_b, loja_c])
    pedidos = [(7, 0), (8, 1), (9, 2)]  # próximos 3 meses após o 6º mês da loja nova
    previsao_nova = projetar_com_sazonal_emprestado(loja_nova, indices_emprestados, pedidos)

    print("=== Cold start (unidade nova, só 6 meses de histórico) ===")
    print("Índices sazonais emprestados das 3 unidades maduras (média = 1.0 por ano):")
    for mes in range(1, 13):
        print(f"  {MESES_PT[mes - 1]}: {indices_emprestados[mes]:.2f}")
    print("\nPrevisão pros próximos 3 meses da unidade nova:")
    for (mes_alvo, passos), valor in previsao_nova.items():
        print(f"  {MESES_PT[mes_alvo - 1]} (+{passos + 1} mês(es) à frente): R$ {valor:,.2f}")

    _plotar(treino, retido, previstos, indices_emprestados)


def _plotar(treino, retido, previstos, indices_emprestados):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib não instalado — pulando geração do gráfico)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    meses_treino = list(range(1, len(treino) + 1))
    meses_retido = list(range(len(treino) + 1, len(treino) + len(retido) + 1))
    ax1.plot(meses_treino, [m["faturamento"] for m in treino], label="Histórico (treino)", color="#4b3b8f")
    ax1.plot(meses_retido, [m["faturamento"] for m in retido], "o-", label="Real (retido)", color="#2b2440")
    ax1.plot(meses_retido, previstos, "o--", label="Previsto (Holt-Winters)", color="#e07a3f")
    ax1.set_title("Backtest — 6 meses retidos")
    ax1.set_xlabel("mês (índice da série)")
    ax1.set_ylabel("faturamento (sintético)")
    ax1.legend(fontsize=8)

    meses = list(range(1, 13))
    ax2.bar(meses, [indices_emprestados[m] for m in meses], color="#8a6ff0")
    ax2.axhline(1.0, color="#999", linewidth=1, linestyle=":")
    ax2.set_xticks(meses)
    ax2.set_xticklabels(MESES_PT, fontsize=8)
    ax2.set_title("Sazonalidade emprestada (média de 3 unidades)")
    ax2.set_ylabel("índice (1.0 = média do ano)")

    fig.tight_layout()
    fig.savefig("demo_grafico.png", dpi=130)
    print("\nGráfico salvo em demo_grafico.png")


if __name__ == "__main__":
    main()
