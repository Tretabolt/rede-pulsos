"""
Homeostase no Exp 3: segurar a separação que o R-STDP com crédito achou
em 16 ensaios e perdeu em 40.

Baseline Exp 3 (treino_memoria_especifica.py, sem homeostase):
  16 ensaios: sol 15.76, calor 9.95, chuva -10.1
    bom 9 VAI, chuva 3 VAI (errado por 1), só sol 6 VAI, só calor 0 FICA
    acertos 3/4, margem 12
  40 ensaios: sol 29.17, calor 27.22, chuva -30.0 (teto)
    bom 19 VAI, chuva 5 VAI, só sol 13 VAI, só calor 5 VAI
    acertos 2/4, margem 22
O problema: sol e calor precedem o pós nos dois contextos, então o
STDP fortalece o "vai" nos dois. Sem freio, os dois saturam em 30 e
a chuva, mesmo no teto -30, não segura o dia de chuva nem o só calor.

Três freios testados, todos sobre o mesmo treino (16 + 24 ensaios,
alternando bom/chuva, recompensa com crédito, rede congelada só na
avaliação final — o treino continua com STDP ligado, como antes):

  H1 — STDP dependente do peso (multiplicativo): o Δw que AUMENTA |w|
       é multiplicado por (30 - |w|)/30. Perto do teto, a potenciação
       some; enfraquecer continua normal. O teto continua 30, mas a
       rede não chega lá — ela assintota.
  H2 — Teto excitatório baixo: w_max = 20 para sol/calor, w_min = -20
       para a chuva. Simples, é um teto, não um mecanismo — mas é o
       que o Dan tinha posto como alternativa ("homeostase ou teto").
  H3 — Normalização: depois de cada ensaio, se sol + calor > 18
       (a soma inicial, 10 + 8), os dois são reescalados para somar 18.
       Soma fixa cria competição: se o sol cresce, o calor tem que
       encolher. A chuva não entra na soma.

Sucesso = em 40 ensaios, acertos >= 3/4 e chuva e só calor em FICA,
sem nenhum peso saturado em 30 (ou -30, no H2, em -20).

Checagem de estabilidade (rodada depois, incluída no main): H1 até
80 ensaios regride parcialmente — sol 23.97, calor 13.42, chuva -19.5,
chuva 4 VAI, acertos 3/4, margem 20 (baseline em 80: 2/4). Ou seja,
H1 segura a separação numa janela (16–40 ensaios, 4/4), mas não é
um fixo permanente: o fator (30-|w|)/30 atrasa a saturação, não a
impede. A memória específica emergiu, mas ainda não é estável para
sempre — esse é o próximo fio aberto, não um resultado escondido.

Rode: python3 homeostase_exp3.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

import matplotlib.pyplot as plt
import numpy as np

from rede_pulsos.tarefa_bike import rodar_cenario
from treino_bike import teste_final
from treino_memoria_especifica import montar_rstdp, pesos

CORRETO = {"bom": True, "chuva": False, "só sol": True, "só calor": False}


def escore(resultado):
    margem, acertos = 0, 0
    for nome, (n, vai) in resultado.items():
        certo = CORRETO[nome]
        margem += (n - 3) if certo else (3 - n)
        if vai == certo:
            acertos += 1
    return margem, acertos


def treinar_credito_homeostase(rede, ensaios=40, modo=None):
    """Como treinar_credito, com um freio por ensaio. modo None/H1/H2/H3.
    H2 é configurado fora, no montar (teto 20); aqui só treina."""
    historico = []
    for i in range(1, ensaios + 1):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            if modo == "H1":
                # STDP dependente do peso: só freia o Δw que aumenta |w|.
                dw_raw = rec * s.elegibilidade
                if dw_raw * s.w > 0:  # aumenta |w|
                    dw_raw *= (30.0 - abs(s.w)) / 30.0
                s.w = float(min(max(s.w + dw_raw, s.w_min), s.w_max))
                s.elegibilidade = 0.0
            else:
                s.aplicar_elegibilidade(rec)
        if modo == "H3":
            exc = [s for s in rede.sinapses if not s.inibitoria]
            total = sum(s.w for s in exc)
            if total > 18.0:
                fator = 18.0 / total
                for s in exc:
                    s.w *= fator
        if i in (16, 40):
            historico.append((i, dict(pesos(rede))))
    return historico


def avaliar(rede, rotulo):
    r = teste_final(rede)
    margem, acertos = escore(r)
    print(f"  {rotulo}: pesos={pesos(rede)}")
    print("    teste: " + "; ".join(f"{c}: {n} {'VAI' if v else 'FICA'}" for c, (n, v) in r.items()))
    print(f"    acertos {acertos}/4, margem {margem}")
    return margem, acertos


def main():
    variantes = [
        ("Baseline (sem freio)", None, None),
        ("H1 — STDP dependente do peso", "H1", None),
        ("H2 — teto excitatório 20", None, "teto20"),
        ("H3 — normalização soma=18", "H3", None),
    ]
    resultados = {}
    for rotulo, modo, setup in variantes:
        rede = montar_rstdp()
        if setup == "teto20":
            for s in rede.sinapses:
                if s.inibitoria:
                    s.w_min = -20.0
                else:
                    s.w_max = 20.0
        print(f"\n== {rotulo} ==")
        hist = treinar_credito_homeostase(rede, ensaios=40, modo=modo)
        for i, p in hist:
            print(f"  após {i} ensaios: pesos={p}")
        margem, acertos = avaliar(rede, "final 40 ensaios")
        resultados[rotulo] = (margem, acertos, dict(pesos(rede)))

    print("\n== Resumo em 40 ensaios ==")
    for rotulo, (margem, acertos, p) in resultados.items():
        print(f"  {rotulo}: acertos {acertos}/4, margem {margem}, pesos={p}")

    print("\n== Checagem aos 16 ensaios: H1 ==")
    rede16 = montar_rstdp()
    treinar_credito_homeostase(rede16, ensaios=16, modo="H1")
    avaliar(rede16, "H1 16 ensaios")

    print("\n== Checagem de estabilidade: H1 até 80 ensaios ==")
    rede80 = montar_rstdp()
    treinar_credito_homeostase(rede80, ensaios=80, modo="H1")
    avaliar(rede80, "H1 80 ensaios")
    from treino_memoria_especifica import treinar_credito
    rede80b = montar_rstdp()
    treinar_credito(rede80b, ensaios=80)
    avaliar(rede80b, "Baseline 80 ensaios")

    # figura: pesos finais por variante
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Homeostase no Exp 3: qual freio segura a separação até 40 ensaios?", fontsize=11, fontweight="bold")
    rotulos = ["Base", "H1", "H2", "H3"]
    x = np.arange(len(rotulos))
    w = 0.25
    sol = [resultados[r]["sol"] if False else list(resultados.values())[i][2]["sol"] for i, r in enumerate(resultados)]
    # pesos dict: sol, calor, chuva
    sols = [list(resultados.values())[i][2]["sol"] for i in range(len(resultados))]
    calores = [list(resultados.values())[i][2]["calor"] for i in range(len(resultados))]
    chuvas = [list(resultados.values())[i][2]["chuva"] for i in range(len(resultados))]
    ax = axes[0]
    ax.bar(x - w, sols, w, label="sol", color="#1a73e8")
    ax.bar(x, calores, w, label="calor", color="#e8710a")
    ax.bar(x + w, chuvas, w, label="chuva", color="#34a853")
    ax.axhline(30, color="gray", linestyle="--", linewidth=1, label="teto 30")
    ax.axhline(-30, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=8)
    ax.set_ylabel("Peso final (40 ensaios)")
    ax.set_title("1. Pesos finais", loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    ax = axes[1]
    margens = [v[0] for v in resultados.values()]
    acertos = [v[1] for v in resultados.values()]
    ax.bar(x - w / 2, margens, w, label="margem", color="#1a73e8")
    ax2 = ax.twinx()
    ax2.plot(x, acertos, "o-", color="#e8710a", linewidth=2, label="acertos /4")
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=8)
    ax.set_ylabel("Margem total")
    ax2.set_ylabel("Acertos (0-4)")
    ax2.set_ylim(0, 4.5)
    ax.set_title("2. Margem e acertos em 40 ensaios", loc="left")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = "/home/hatch/workspace/rede-pulsos/homeostase_exp3.png"
    plt.savefig(out, dpi=200)
    print(f"Gráfico salvo em {out}")


if __name__ == "__main__":
    main()
