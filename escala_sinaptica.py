"""Escala sináptica — o homeostato de rede que faltava testar.

Fio aberto da conclusão de 24/11: a quantização dá memória, o veto
dá permanência, e o que faz o sol fugir para o teto (22.5 -> 30 entre
16 e 40 ensaios, e o dia de chuva vira 5 VAI) é a falta de uma noção
de proporção — cada sinapse aprende com a própria elegibilidade e
nada mantém a excitação em proporção com a inibição. O candidato é
a escala sináptica homeostática (Turrigiano): o neurônio multiplica
todos os seus pesos excitatórios para manter a própria taxa de
disparo num alvo. Ela age no nível da taxa da rede — diferente do
degrau (atualização) e do veto (estrutura) — então tem chance real
de compor com os dois.

Regra (S), aplicada no fim de cada ensaio, depois do R-STDP:
  n = nº de disparos da decisão no ensaio
  T = 6 no dia bom, 1 no dia de chuva (o alvo é a taxa do contexto)
  fator = 1 + g·(T − n)/T, g = 0.1, preso em [0.9, 1.1]
  pesos excitatórios (sol, calor) *= fator, dentro de [0, 30]
A inibitória não é escalada: a razão E/I se move porque só um lado
muda. Quando o neurônio dispara demais, a excitação encolhe; quando
dispara de menos, ela cresce. O alvo 6 é a taxa do dia bom no início
(5 VAI); o alvo 1 no dia de chuva é "quase silêncio".

Modos (todos Exp 3 com crédito, mesmo início 10/8/-3, 16/40/80):
  S-contínuo: escala direta no peso contínuo.
  S-quant:    quantização simples 3.75 + escala direta (snap depois).
  S-Q2:       resíduo Q2 + escala via resíduo: a mudança multiplicativa
              vira dw_esc = w·(fator−1) somado ao resíduo da sinapse —
              nada se perde, só se atrasa, como no Q2 puro.

Pergunta: a escala impede a fuga do sol e faz a quantização segurar
4/4 até 80 ensaios sem veto? Ou ela briga com o degrau como o H1
brigou (o freio encolhe os Δw para baixo de meio degrau)?

Rode: python3 escala_sinaptica.py
"""
import math
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.tarefa_bike import rodar_cenario
from treino_bike import teste_final
from treino_memoria_especifica import montar_rstdp
from arestas_validas import ablacao, congelar, escore_acerto
from pesos_quantizados import quantizar

UNIDADE = 3.75
GANHO = 0.1
ALVO_BOM = 6
ALVO_CHUVA = 1
FATOR_MIN, FATOR_MAX = 0.9, 1.1


def pesos(rede):
    return {s.pre.nome: round(s.w, 2) for s in rede.sinapses}


def fator_escala(n, bom):
    """Fator multiplicativo da escala para este ensaio."""
    alvo = ALVO_BOM if bom else ALVO_CHUVA
    f = 1.0 + GANHO * (alvo - n) / alvo
    return max(FATOR_MIN, min(FATOR_MAX, f))


def _n_degraus(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def treinar_escala(rede, inicio, fim, modo):
    """Exp 3 com crédito + escala. modo: 'continuo', 'quant' ou 'q2'.
    O resíduo do Q2 vive em rede._residuo (dict por sinapse), criado
    aqui na primeira chamada, ancorado nos pesos iniciais."""
    if modo == "q2" and not hasattr(rede, "_residuo"):
        rede._residuo = {id(s): 0.0 for s in rede.sinapses}
        rede._ancora = {id(s): s.w for s in rede.sinapses}
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
            dw = rec * s.elegibilidade
            s.elegibilidade = 0.0
            if modo == "q2":
                rede._residuo[id(s)] += dw
                if not s.inibitoria:
                    rede._residuo[id(s)] += s.w * (f - 1.0)
                n = _n_degraus(rede._residuo[id(s)] / UNIDADE)
                if n != 0:
                    novo = rede._ancora[id(s)] + (
                        _n_degraus((s.w - rede._ancora[id(s)]) / UNIDADE)
                        + n
                    ) * UNIDADE
                    s.w = float(min(max(novo, s.w_min), s.w_max))
                    rede._residuo[id(s)] -= n * UNIDADE
            else:
                s.w = float(min(max(s.w + dw, s.w_min), s.w_max))
                if not s.inibitoria:
                    s.w = float(min(max(s.w * f, s.w_min), s.w_max))
                if modo == "quant":
                    s.w = float(min(max(quantizar(s.w, UNIDADE),
                                        s.w_min), s.w_max))


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
    for modo, rot in [("continuo", "S-contínuo"),
                      ("quant", "S-quant 3.75"),
                      ("q2", "S-Q2")]:
        print(f"\n== {rot} (ganho {GANHO}, alvos bom {ALVO_BOM} / "
              f"chuva {ALVO_CHUVA}) ==")
        rede = montar_rstdp()
        treinar_escala(rede, 1, 16, modo)
        a16 = avaliar(rede, "16 ensaios")
        treinar_escala(rede, 17, 40, modo)
        a40 = avaliar(rede, "40 ensaios")
        treinar_escala(rede, 41, 80, modo)
        a80 = avaliar(rede, "80 ensaios")
        resultados[rot] = (a16, a40, a80)

    print("\n== Comparação (acertos/4 | margem) ==")
    for rot, (a16, a40, a80) in resultados.items():
        print(f"  {rot:12s} 16: {a16[0]}/4 | {a16[1]:3d}   "
              f"40: {a40[0]}/4 | {a40[1]:3d}   80: {a80[0]}/4 | {a80[1]:3d}")
    print("  (referência: contínuo 3/4 | 12, 2/4 | 22, 2/4 | 21; "
          "quant 3.75 4/4 | 17, 3/4 | 23, 3/4 | 23; "
          "Q2 4/4 | 12, 2/4 | 17, 2/4 | 22; "
          "Q2+T1 4/4 | 12, 4/4 | 26, 4/4 | 26)")


if __name__ == "__main__":
    main()
