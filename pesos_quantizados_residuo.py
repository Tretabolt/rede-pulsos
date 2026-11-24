"""Pesos quantizados com resíduo — a tentativa de conserto.

Fio aberto do teste anterior (pesos_quantizados_homeostase.py):
H1 + quantização 3.75 congela a rede em 2/4 — o freio H1 encolhe
os Δw por ensaio e eles nunca cruzam meio degrau, então a escada
descarta tudo. A quantização simples joga fora o que não cruza o
degrau; este teste guarda o resto.

Q2 — quantização com resíduo (sigma-delta): cada sinapse acumula
num resíduo o Δw de cada ensaio (com o freio H1, quando ligado);
só quando o resíduo acumulado cruza meio degrau é que o peso salta
um número inteiro de degraus, e o resto fica guardado para o
próximo ensaio. Nada se perde, só se atrasa. A escada é ancorada
nos pesos iniciais (10, 8, -3): o peso vale sempre w0 + k·unidade,
com k inteiro — discreto, mas partindo do mesmo início comum que
todos os outros testes.

Pergunta: o resíduo destrava a combinação H1 + 3.75? E sozinho,
ele melhora a quantização pura em 40/80 ensaios?

Rode: python3 pesos_quantizados_residuo.py
"""
import math
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.tarefa_bike import rodar_cenario
from treino_bike import teste_final
from treino_memoria_especifica import montar_rstdp
from arestas_validas import ablacao, congelar, escore_acerto

UNIDADE = 3.75


def pesos(rede):
    return {s.pre.nome: round(s.w, 2) for s in rede.sinapses}


def _n_degraus(x):
    """Inteiro mais próximo, meio para longe de zero."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def treinar_q2(rede, inicio, fim, modo=None):
    """Treina os ensaios [inicio, fim] com resíduo. O estado do
    resíduo vive em rede._residuo (dict por sinapse), criado aqui
    na primeira chamada."""
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
            if modo == "H1":
                dw = rec * s.elegibilidade
                if dw * s.w > 0:
                    dw *= (30.0 - abs(s.w)) / 30.0
                s.elegibilidade = 0.0
            else:
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


def avaliar(rede, rotulo):
    flags = [s.stdp for s in rede.sinapses]
    congelar(rede)
    r = teste_final(rede)
    margem, acertos = escore_acerto(r)
    ab = ablacao(rede)
    nocivas = [e for e, (_, d) in ab.items() if d > 0]
    for s, f in zip(rede.sinapses, flags):
        s.stdp = f
    print(f"  {rotulo}: pesos={pesos(rede)}")
    print("    teste: " + "; ".join(
        f"{c}: {n} {'VAI' if v else 'FICA'}" for c, (n, v) in r.items()))
    print(f"    acertos {acertos}/4, margem {margem} | nocivas {nocivas}")
    return acertos, margem


def main():
    resultados = {}
    for modo in [None, "H1"]:
        rot = "Q2 + H1" if modo else "Q2 sozinho"
        print(f"\n== {rot} (unidade 3.75, com resíduo) ==")
        rede = montar_rstdp()
        treinar_q2(rede, 1, 16, modo)
        a16 = avaliar(rede, "16 ensaios")
        treinar_q2(rede, 17, 40, modo)
        a40 = avaliar(rede, "40 ensaios")
        treinar_q2(rede, 41, 80, modo)
        a80 = avaliar(rede, "80 ensaios")
        resultados[rot] = (a16, a40, a80)

    print("\n== Comparação (acertos/4 | margem) ==")
    for rot, (a16, a40, a80) in resultados.items():
        print(f"  {rot:12s} 16: {a16[0]}/4 | {a16[1]:3d}   "
              f"40: {a40[0]}/4 | {a40[1]:3d}   80: {a80[0]}/4 | {a80[1]:3d}")
    print("  (referência: quantização simples 3.75: 4/4 | 17, 3/4 | 23, "
          "3/4 | 23; H1 sozinho: 4/4 | 9, 4/4 | 15, 3/4 | 20)")


if __name__ == "__main__":
    main()
