"""Grafo maior v0.2 + escala sináptica.

Fio aberto do v0.2: a rede entrou num regime hiper-excitado (a
decisão dispara 40 vezes no dia bom), a regra "3 ou mais = VAI"
perdeu a discriminação, o dia de chuva falha, e o calor foi
calado pela lateral sol->calor (0 disparos, arestas inertes).
A escala sináptica (a mesma de escala_sinaptica.py: fator
multiplicativo sobre os pesos excitatórios, alvo 6 disparos no dia
bom e 1 no dia de chuva, ganho 0,1, preso em [0,9, 1,1]) é o
freio com princípio para esse regime: ela puxa a taxa de volta
para a faixa onde o limiar de 3 discrimina.

Treino: Exp 3 com crédito + escala no fim de cada ensaio, 16/40/80.
A escala só toca sinapses excitatórias; as inibitórias (chuva e
laterais) e o feedback fixo não são escalados. Mesma avaliação do
v0.1/v0.2: rede congelada, 4 contextos, ablação por par,
ISC_Local = válidos / 15.

Perguntas: a escala doma as 40 contagens e devolve o dia de
chuva? O calor volta a disparar (a lateral sol->calor enfraquece
com o STDP) ou continua calado? O ISC sai de 0,60?

Rode: python3 grafo_maior_v2_escala.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.tarefa_bike import rodar_cenario
from grafo_maior import avaliar, diagnostico_inicial
from grafo_maior_v2 import montar_grafo_v2

GANHO = 0.1
ALVO_BOM = 6
ALVO_CHUVA = 1
FATOR_MIN, FATOR_MAX = 0.9, 1.1


def pesos(rede):
    return {f"{s.pre.nome}->{s.pos.nome}": round(s.w, 2)
            for s in rede.sinapses}


def fator_escala(n, bom):
    alvo = ALVO_BOM if bom else ALVO_CHUVA
    f = 1.0 + GANHO * (alvo - n) / alvo
    return max(FATOR_MIN, min(FATOR_MAX, f))


def treinar_escala(rede, inicio, fim):
    for i in range(inicio, fim + 1):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        f = fator_escala(r["n_decisao"], bom)
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            s.aplicar_elegibilidade(rec)
            if not s.inibitoria:
                s.w = float(min(max(s.w * f, s.w_min), s.w_max))


def main():
    print("== Grafo maior v0.2 + escala (ganho 0.1, alvos 6/1) ==")
    rede = montar_grafo_v2()
    print(f"  início: {pesos(rede)}")
    diagnostico_inicial(rede)
    treinar_escala(rede, 1, 16)
    print(f"  pesos aos 16: {pesos(rede)}")
    a16 = avaliar(rede, "16 ensaios")
    treinar_escala(rede, 17, 40)
    print(f"  pesos aos 40: {pesos(rede)}")
    a40 = avaliar(rede, "40 ensaios")
    treinar_escala(rede, 41, 80)
    print(f"  pesos aos 80: {pesos(rede)}")
    a80 = avaliar(rede, "80 ensaios")
    print("\n== Resumo (acertos/4 | margem | ISC_Local) ==")
    for rot, a in [("16", a16), ("40", a40), ("80", a80)]:
        print(f"  {rot} ensaios: {a[0]}/4 | {a[1]:3d} | {a[2]:.3f}")
    print("  (referência v0.2 sem escala: 3/4 | 73, 3/4 | 61, 3/4 | 63; "
          "ISC 0,667 / 0,600 / 0,600)")


if __name__ == "__main__":
    main()
