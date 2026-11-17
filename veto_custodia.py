"""
Veto por custódia: bloquear uma aresta só depois de medir o que ela faz,
não por causa do rótulo dela.

Definição v1.0 (arestas_validas.py, C3): uma aresta é NOCIVA se zerá-la
MELHORA o escore da rede; válida se zerá-la PIORA; inerte se nada muda.
O escore é a margem total nos 4 contextos (bom VAI, chuva FICA,
só sol VAI, só calor FICA) + nº de acertos.

Veto por custódia: para cada estado treinado e congelado, abla cada
aresta (mede, restaura), veta — zera de vez — só as nocivas, e reavalia.
A rede é congelada (stdp=False): medir não treina, e o veto não treina.

Contraste — veto por rótulo: veta a aresta de maior |w|, com a regra
ingênua "peso grande = atalho". No MNHI 3.0 / harness, o análogo é o
veto que depende do rótulo "ATALHO" na pista, não de custódia medida.
Aqui o maior peso é quase sempre o sol — e a ablação mostra que o sol
é a aresta mais válida (delta -13 a -22). O rótulo pune a melhor aresta.

Limites honestos, já no código:
  1. O escore (o que é VAI/FICA certo) vem de fora, dado por nós.
     A custódia verifica a aresta, mas não descobre sozinha o que é certo.
  2. Cada estado aqui tem no máximo 1 aresta nociva, sempre o calor.
     Vetar todas as nocivas de uma vez é exato aqui; num grafo maior,
     com várias nocivas, a interação entre elas exigiria veto iterativo.
  3. Aresta inerte (Exp3-16, calor delta 0) é mantida. O veto melhora
     o escore, não o ISC: o ISC C3 fica em 0,33, porque o veto tira
     uma aresta e não cria nenhuma válida.
  4. O veto é destrutivo (w=0) e só foi testado nos 4 contextos do
     escore. Fora deles, a aresta vetada podia carregar informação.

Rode: python3 veto_custodia.py
"""
import copy
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

import matplotlib.pyplot as plt
import numpy as np

from arestas_validas import construir_estados, escore_acerto, ablacao, pesos
from treino_bike import teste_final


def aplicar_veto(rede, alvos):
    """Zera de vez as arestas em alvos. Devolve os pesos finais."""
    for s in rede.sinapses:
        if s.pre.nome in alvos:
            s.w = 0.0
    return pesos(rede)


def main():
    estados = construir_estados()
    nomes = list(estados.keys())
    dados = {}

    for nome in nomes:
        base_rede = estados[nome]
        # Duas cópias idênticas do mesmo estado congelado: uma para cada veto.
        rede_custodia = copy.deepcopy(base_rede)
        rede_rotulo = copy.deepcopy(base_rede)

        base = teste_final(base_rede)
        base_margem, base_acertos = escore_acerto(base)
        ab = ablacao(base_rede)
        nocivas = [e for e, (_, dl) in ab.items() if dl > 0]
        validas = [e for e, (_, dl) in ab.items() if dl < 0]
        inertes = [e for e, (_, dl) in ab.items() if dl == 0]

        # Custódia: veta só as nocivas medidas.
        aplicar_veto(rede_custodia, nocivas)
        r_c = teste_final(rede_custodia)
        margem_c, acertos_c = escore_acerto(r_c)

        # Rótulo: veta a de maior |w| ("peso grande = atalho").
        maior = max(base_rede.sinapses, key=lambda s: abs(s.w)).pre.nome
        aplicar_veto(rede_rotulo, [maior])
        r_r = teste_final(rede_rotulo)
        margem_r, acertos_r = escore_acerto(r_r)

        dados[nome] = {
            "pesos": pesos(base_rede), "base": (base_margem, base_acertos),
            "ablacao": ab, "nocivas": nocivas, "validas": validas,
            "inertes": inertes, "custodia": (margem_c, acertos_c, r_c),
            "rotulo_alvo": maior, "rotulo": (margem_r, acertos_r, r_r),
        }

    print("== Veto por custódia vs veto por rótulo ==")
    print("Base = margem total / acertos em 4 contextos.")
    for nome, d in dados.items():
        bm, ba = d["base"]
        cm, ca, _ = d["custodia"]
        rm, ra, _ = d["rotulo"]
        print(f"{nome}: pesos={d['pesos']}")
        print(f"   ablação: " + "; ".join(f"{e}: {dl:+.0f}" for e, (_, dl) in d["ablacao"].items())
              + f" | nocivas {d['nocivas']} | inertes {d['inertes']}")
        print(f"   base: margem {bm}, acertos {ba}/4")
        print(f"   custódia (veta {d['nocivas'] or 'nada'}): margem {cm} ({cm-bm:+d}), acertos {ca}/4")
        print(f"   rótulo (veta maior |w| = {d['rotulo_alvo']}): margem {rm} ({rm-bm:+d}), acertos {ra}/4")

    n_melhorou = sum(1 for d in dados.values() if d["custodia"][0] > d["base"][0])
    n_acertos_subiu = sum(1 for d in dados.values() if d["custodia"][1] > d["base"][1])
    n_rotulo_piorou = sum(1 for d in dados.values() if d["rotulo"][0] < d["base"][0])
    print(f"\nResumo: custódia melhora a margem em {n_melhorou}/7 estados "
          f"e os acertos em {n_acertos_subiu}/7; nunca piora. "
          f"Rótulo piora a margem em {n_rotulo_piorou}/7 estados.")

    # figura: margem e acertos, base vs custódia vs rótulo
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Veto por custódia melhora; veto por rótulo pune a melhor aresta", fontsize=11, fontweight="bold")
    x = np.arange(len(nomes))
    w = 0.25
    base_m = [d["base"][0] for d in dados.values()]
    cust_m = [d["custodia"][0] for d in dados.values()]
    rot_m = [d["rotulo"][0] for d in dados.values()]
    ax = axes[0]
    ax.bar(x - w, base_m, w, label="Base", color="#9aa0a6")
    ax.bar(x, cust_m, w, label="Custódia", color="#1a73e8")
    ax.bar(x + w, rot_m, w, label="Rótulo (maior |w|)", color="#e8710a")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ensaios", "") for n in nomes], rotation=30, ha="right")
    ax.set_ylabel("Margem total do escore")
    ax.set_title("1. Margem por estado", loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    base_a = [d["base"][1] for d in dados.values()]
    cust_a = [d["custodia"][1] for d in dados.values()]
    rot_a = [d["rotulo"][1] for d in dados.values()]
    ax = axes[1]
    ax.bar(x - w, base_a, w, label="Base", color="#9aa0a6")
    ax.bar(x, cust_a, w, label="Custódia", color="#1a73e8")
    ax.bar(x + w, rot_a, w, label="Rótulo (maior |w|)", color="#e8710a")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ensaios", "") for n in nomes], rotation=30, ha="right")
    ax.set_ylabel("Acertos em 4 contextos")
    ax.set_ylim(0, 4.5)
    ax.set_title("2. Acertos por estado", loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = "/home/hatch/workspace/rede-pulsos/veto_custodia.png"
    plt.savefig(out, dpi=200)
    print(f"Gráfico salvo em {out}")


if __name__ == "__main__":
    main()
