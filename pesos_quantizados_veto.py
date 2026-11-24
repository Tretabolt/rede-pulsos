"""Q2 + T1: pesos quantizados com resíduo + veto por custódia durante
o treino — a combinação que faltava.

Fio aberto do trabalho de 23/11 (pesos quantizados):
  - Quantização simples 3.75: 4/4 aos 16 ensaios, 3/4 aos 40/80.
  - Q2 (resíduo sigma-delta): 4/4 aos 16, mas COLAPSA para 2/4 aos
    40/80 — o resíduo destrava o calor, que satura em 30 e o contexto
    só-calor vira VAI (errado). O freio da zona morta era acidental.
  - T1 (veto por custódia a cada 8 ensaios, sem quantização): o veto
    dispara uma vez, no ensaio 32 (calor delta +4), poda o calor para
    0 e congela; 4/4 aos 40 e aos 80 — o único 4/4 aos 80 até agora.

Pergunta: o veto durante o treino salva a quantização? Se o calor
quantizado for podado antes de saturar, a rede pode segurar 4/4 até
80 ensaios — ou o veto nunca dispara (o calor quantizado pode não
parecer nocivo no momento da checagem), ou dispara tarde demais.

Combinação (este script):
  - Exp 3, R-STDP com atribuição de crédito (treino_memoria_especifica).
  - Pesos quantizados com resíduo Q2, unidade 3.75, escada ancorada
    nos pesos iniciais (10, 8, -3) — pesos_quantizados_residuo.
  - Veto por custódia a cada 8 ensaios, escore de ablação nos 4
    contextos, rede congelada na medição — veto_durante_treino (T1).

Dois cuidados de interação, registrados aqui:
  1. Aresta vetada (w=0, stdp=False) não acumula elegibilidade, então
     o resíduo dela nunca mais cruza um degrau — não há regrow. Por
     segurança, o resíduo da aresta vetada é zerado na hora do veto.
  2. O veto usa os 4 contextos no escore de ablação, incluindo só sol
     e só calor, que não são ensaios de treino — o mesmo vazamento
     documentado em veto_durante_treino.py (T3). Aqui não há variante
     T3; o vazamento vale igual para T1 e para este teste, então a
     comparação é justa.

Protocolo comum a todas as corridas: pesos iniciais sol 10 / calor 8
/ chuva -3, dias bons e chuvosos alternados, avaliação congelada
(stdp=False) aos 16, 40 e 80 ensaios nos 4 contextos.

Rode: python3 pesos_quantizados_veto.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.tarefa_bike import rodar_cenario
from treino_memoria_especifica import montar_rstdp
from arestas_validas import ablacao, congelar
from pesos_quantizados_residuo import (
    UNIDADE,
    _n_degraus,
    avaliar,
    pesos,
)

VETO_CADA = 8


def treinar_q2_veto(rede, inicio, fim, eventos):
    """Treina os ensaios [inicio, fim]: Q2 por ensaio + veto a cada
    8 ensaios. Os eventos de veto vão para a lista `eventos` como
    (ensaio, deltas_de_ablação, arestas_vetadas, pesos_após_veto)."""
    if not hasattr(rede, "_residuo"):
        rede._residuo = {id(s): 0.0 for s in rede.sinapses}
        rede._ancora = {id(s): s.w for s in rede.sinapses}
    for i in range(inicio, fim + 1):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            dw = rec * s.elegibilidade
            s.elegibilidade = 0.0
            rede._residuo[id(s)] += dw
            n = _n_degraus(rede._residuo[id(s)] / UNIDADE)
            if n != 0:
                novo = rede._ancora[id(s)] + (
                    _n_degraus((s.w - rede._ancora[id(s)]) / UNIDADE) + n
                ) * UNIDADE
                s.w = float(min(max(novo, s.w_min), s.w_max))
                rede._residuo[id(s)] -= n * UNIDADE
        if i % VETO_CADA == 0:
            # Veto por custódia, rede congelada: medir não pode treinar.
            flags = [s.stdp for s in rede.sinapses]
            congelar(rede)
            ab = ablacao(rede)
            deltas = {e: d for e, (_, d) in ab.items()}
            for s, f in zip(rede.sinapses, flags):
                s.stdp = f
            nocivas = [e for e, d in deltas.items() if d > 0]
            for s in rede.sinapses:
                if s.pre.nome in nocivas:
                    s.w = 0.0
                    s.stdp = False  # poda + congelamento: não regrow
                    rede._residuo[id(s)] = 0.0
            eventos.append((i, deltas, list(nocivas), dict(pesos(rede))))


def main():
    print("== Q2 + T1: quantização 3.75 com resíduo + veto a cada 8 ==")
    rede = montar_rstdp()
    eventos = []
    print(f"início: {pesos(rede)}")
    treinar_q2_veto(rede, 1, 16, eventos)
    a16 = avaliar(rede, "16 ensaios")
    treinar_q2_veto(rede, 17, 40, eventos)
    a40 = avaliar(rede, "40 ensaios")
    treinar_q2_veto(rede, 41, 80, eventos)
    a80 = avaliar(rede, "80 ensaios")

    print("\n== Eventos de veto (cheque a cada 8 ensaios) ==")
    for i, deltas, nocivas, p in eventos:
        print(f"  ensaio {i:2d}: deltas " +
              "; ".join(f"{e}: {d:+.0f}" for e, d in deltas.items()) +
              f" -> vetadas {nocivas or 'nenhuma'} pesos={p}")

    print("\n== Comparação (acertos/4 | margem) ==")
    for rot, a in [("Q2 + T1", (a16, a40, a80))]:
        print(f"  {rot:12s} 16: {a[0][0]}/4 | {a[0][1]:3d}   "
              f"40: {a[1][0]}/4 | {a[1][1]:3d}   80: {a[2][0]}/4 | {a[2][1]:3d}")
    print("  (referência: Q2 sozinho: 4/4 | 12, 2/4 | 17, 2/4 | 22; "
          "T1 sozinho: 4/4 | 26 aos 40 e 80; quant simples 3.75: "
          "4/4 | 17, 3/4 | 23, 3/4 | 23)")


if __name__ == "__main__":
    main()
