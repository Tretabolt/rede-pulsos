"""
Veto durante o treino: medir as arestas no meio do aprendizado e vetar
a nociva antes dela arrastar o treino — em vez de só limpar depois.

Base: Exp 3, R-STDP com atribuição de crédito, 80 ensaios alternando
bom/chuva. A cada 8 ensaios, pausa o treino e faz o veto por custódia:
abla cada aresta (com a rede CONGELADA — medir não pode treinar),
veta de vez só as nocivas (ablação melhora o escore), e congela a
aresta vetada (stdp=False, w=0): veto é poda + congelamento, não um
reset temporário, senão o STDP regrow a aresta no ensaio seguinte.

Variantes:
  T0 baseline Exp 3 (sem veto, sem H1) — referência, já conhecida.
  T1 Exp 3 + veto a cada 8 ensaios, escore de ablação nos 4 contextos.
  T2 Exp 3 + H1 (STDP dependente do peso) + veto a cada 8 ensaios.
  T3 Exp 3 + veto a cada 8 ensaios, mas o escore de ablação usa SÓ
     bom+chuva (os contextos de treino). Os outros dois contextos
     (só sol, só calor) ficam de fora da decisão do veto.

Por que T3 existe — o cuidado honesto deste script: o veto por
custódia precisa de um escore para medir. Se o escore usa os 4
contextos, ele usa também só sol e só calor, que NÃO são ensaios de
treino — são contextos de teste. Isso é vazamento do teste para o
treino, do mesmo tipo da recompensa do R-STDP (que já usa o rótulo
certo do dia), mas mais amplo: o veto enxerga contextos que o treino
nunca viu. T3 mede o quanto o veto perde quando só pode usar o que o
treino vê. Se T1 acerta e T3 não veta nada, o veto durante o treino
depende do vazamento — e isso tem que ficar escrito, não escondido.

Sucesso = em 40 e em 80 ensaios, acertos >= 3/4, sem peso saturado.

Rode: python3 veto_durante_treino.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

import matplotlib.pyplot as plt
import numpy as np

from rede_pulsos.tarefa_bike import rodar_cenario
from treino_bike import teste_final
from treino_memoria_especifica import montar_rstdp, pesos
from homeostase_exp3 import escore

CORRETO_4 = {"bom": True, "chuva": False, "só sol": True, "só calor": False}
CORRETO_2 = {"bom": True, "chuva": False}


def escore_2(resultado):
    margem, acertos = 0, 0
    for nome in ("bom", "chuva"):
        n, vai = resultado[nome]
        certo = CORRETO_2[nome]
        margem += (n - 3) if certo else (3 - n)
        if vai == certo:
            acertos += 1
    return margem, acertos


def avaliar_congelado(rede):
    """teste_final com STDP desligado e restaurado: medir não treina."""
    flags = [s.stdp for s in rede.sinapses]
    for s in rede.sinapses:
        s.stdp = False
    try:
        return teste_final(rede)
    finally:
        for s, f in zip(rede.sinapses, flags):
            s.stdp = f


def ablacao_congelada(rede, so_treino=False):
    """Delta de escore ao zerar cada aresta, rede congelada."""
    base = avaliar_congelado(rede)
    base_m, _ = (escore_2(base) if so_treino else escore(base))
    out = {}
    for s in rede.sinapses:
        w0 = s.w
        s.w = 0.0
        r = avaliar_congelado(rede)
        s.w = w0
        m, _ = (escore_2(r) if so_treino else escore(r))
        out[s.pre.nome] = round(m - base_m, 2)
    return out


def treinar_com_veto(ensaios=80, h1=False, veto_cada=8, so_treino=False):
    rede = montar_rstdp()
    eventos = []
    for i in range(1, ensaios + 1):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            if h1:
                dw_raw = rec * s.elegibilidade
                if dw_raw * s.w > 0:
                    dw_raw *= (30.0 - abs(s.w)) / 30.0
                s.w = float(min(max(s.w + dw_raw, s.w_min), s.w_max))
                s.elegibilidade = 0.0
            else:
                s.aplicar_elegibilidade(rec)
        if veto_cada and i % veto_cada == 0:
            ab = ablacao_congelada(rede, so_treino=so_treino)
            nocivas = [e for e, dl in ab.items() if dl > 0]
            for s in rede.sinapses:
                if s.pre.nome in nocivas:
                    s.w = 0.0
                    s.stdp = False  # poda + congelamento: não regrow
            eventos.append((i, dict(ab), list(nocivas), dict(pesos(rede))))
    return rede, eventos


def relatar(rotulo, rede):
    r = avaliar_congelado(rede)
    margem, acertos = escore(r)
    print(f"  {rotulo}: pesos={pesos(rede)}")
    print("    teste: " + "; ".join(f"{c}: {n} {'VAI' if v else 'FICA'}" for c, (n, v) in r.items()))
    print(f"    acertos {acertos}/4, margem {margem}")
    return margem, acertos


def main():
    variantes = [
        ("T0 baseline Exp 3", dict(h1=False, veto_cada=0)),
        ("T1 Exp 3 + veto a cada 8", dict(h1=False, veto_cada=8)),
        ("T2 Exp 3 + H1 + veto a cada 8", dict(h1=True, veto_cada=8)),
        ("T3 Exp 3 + veto a cada 8 (só bom+chuva)", dict(h1=False, veto_cada=8, so_treino=True)),
    ]
    resultados = {}
    for rotulo, kw in variantes:
        print(f"\n== {rotulo} ==")
        # treina em duas fases para relatar 40 e 80 sem re-treinar do zero:
        # aqui treina direto até 80 e relata o final; o 40 sai do histórico
        # de uma segunda corrida idêntica até 40 (determinístico).
        for fim in (40, 80):
            rede, eventos = treinar_com_veto(ensaios=fim, **kw)
            print(f"  -- até {fim} ensaios --")
            for i, ab, nocivas, p in eventos:
                print(f"    veto no ensaio {i}: deltas " +
                      "; ".join(f"{e}: {dl:+.0f}" for e, dl in ab.items()) +
                      f" -> vetadas {nocivas or 'nenhuma'} pesos={p}")
            margem, acertos = relatar(f"final {fim} ensaios", rede)
            resultados[(rotulo, fim)] = (margem, acertos, dict(pesos(rede)))

    print("\n== Resumo ==")
    for (rotulo, fim), (margem, acertos, p) in resultados.items():
        print(f"  {rotulo} — {fim} ensaios: acertos {acertos}/4, margem {margem}, pesos={p}")

    # figura: acertos por variante, 40 e 80 ensaios
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("Veto durante o treino: segura 4/4 até 80 ensaios?", fontsize=11, fontweight="bold")
    rotulos = ["T0\nbaseline", "T1\nveto 4 ctx", "T2\nH1 + veto", "T3\nveto 2 ctx"]
    x = np.arange(len(rotulos))
    w = 0.35
    a40 = [resultados[(r, 40)][1] for r, _ in variantes]
    a80 = [resultados[(r, 80)][1] for r, _ in variantes]
    ax.bar(x - w / 2, a40, w, label="40 ensaios", color="#1a73e8")
    ax.bar(x + w / 2, a80, w, label="80 ensaios", color="#e8710a")
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos)
    ax.set_ylabel("Acertos em 4 contextos")
    ax.set_ylim(0, 4.5)
    ax.set_title("Acertos por variante", loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = "/home/hatch/workspace/rede-pulsos/veto_durante_treino.png"
    plt.savefig(out, dpi=200)
    print(f"Gráfico salvo em {out}")


if __name__ == "__main__":
    main()
