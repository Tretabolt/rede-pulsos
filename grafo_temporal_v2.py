"""Grafo temporal v0.2 — o laço nasce acima do limiar.

Fio aberto do v0.1: atraso com zero disparos em todos os contextos,
ISC 0,000. O mecanismo estava verificado: um pulso de ~12 mV (laço
w=8) não acorda o parceiro a partir do repouso (~15 mV), e o treino
não cruza o limiar — a recompensa binária pune todo dia bom e
corrói o laço (8 -> 6,8). Armadilha de bootstrap, mesma família da
zona morta da quantização.

Única mudança do v0.2: o laço h1<->h2 nasce em 22, acima do limiar
confirmado (entre 14 e 20). Nada de escala, veto ou moldagem de
recompensa — v0.2 testa a condição inicial, não outro mecanismo.

Perguntas: com o laço acima do limiar, a rede sustenta a resposta
no atraso e aprende 4/4? Os pares recorrentes viram válidos (o
laço importa de verdade) e o ISC sobe em direção a 0,85? E a
discriminação: o laço acende também na chuva e no calor sozinho
(onde deveria ficar quieto), ou a pista fraca não o acende?

Rode: python3 grafo_temporal_v2.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from grafo_maior_v2 import montar_grafo_v2, pesos
from grafo_temporal import (treinar_temporal, avaliar_temporal,
                            diagnostico_inicial)

LACO_INICIAL = 22.0


def montar_temporal_v2():
    rede = montar_grafo_v2()
    for s in rede.sinapses:
        if {s.pre.nome, s.pos.nome} == {"h1", "h2"}:
            s.w = LACO_INICIAL
    return rede


def main():
    print("== Grafo temporal v0.2 (laço h1<->h2 em 22) ==")
    rede = montar_temporal_v2()
    print(f"  início: {pesos(rede)}")
    diagnostico_inicial(rede)
    treinar_temporal(rede, 1, 16)
    a16 = avaliar_temporal(rede, "16 ensaios")
    treinar_temporal(rede, 17, 40)
    a40 = avaliar_temporal(rede, "40 ensaios")
    treinar_temporal(rede, 41, 80)
    a80 = avaliar_temporal(rede, "80 ensaios")
    print("\n== Resumo (acertos/4 | margem | ISC_Local) ==")
    for rot, a in [("16", a16), ("40", a40), ("80", a80)]:
        print(f"  {rot} ensaios: {a[0]}/4 | {a[1]:3d} | {a[2]:.3f}")
    print(f"  pesos aos 80: {pesos(rede)}")


if __name__ == "__main__":
    main()
