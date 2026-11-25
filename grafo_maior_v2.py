"""Grafo maior v0.2 — laterais + calor modulador.

Fio aberto do v0.1: teto de ISC 0,80 (12 pares) e o calor nocivo em
todas as arestas, piorando com o treino — a recorrência amplificou
o problema da estrela. Duas mudanças, testadas juntas porque são o
mesmo redesenho (o papel do calor + as conexões que faltavam):

1. Laterais entre sensores (3 pares novos, teto de ISC 15/15 = 1,0):
   chuva -> sol (-6) e chuva -> calor (-6), inibitórias: a chuva
   veta o sinal de "vai" já na fonte, antes de chegar à camada
   oculta. sol <-> calor (-4 cada direção), inibitórias: competição
   lateral entre as duas entradas excitatórias — afia, não amplifica.
   (A direção calor -> sol excitatória foi descartada de propósito:
   ela faria o calor sozinho acordar o sol e pioraria o contexto
   "só calor".)

2. Calor modulador, não motor: calor -> h1/h2 começa em 6 (metade
   do sol) com teto w_max = 10; calor -> decisão começa em 3 com
   teto 5. O calor pode apoiar, nunca dirigir. É uma hipótese de
   desenho — papéis funcionais diferentes ganham tetos diferentes,
   como entradas moduladoras vs motoras na biologia — e o teste
   (a ablação C3) é que decide se as arestas do calor viram válidas
   ou continuam nocivas. O sol mantém teto 30.

O resto é o v0.1: 6 nós, h1<->h2 recorrente, decisão -> h1
inibitória fixa (-5), Exp 3 com crédito, 16/40/80 ensaios,
avaliação congelada, ablação por par, ISC_Local = válidos / 15.

Perguntas: o ISC chega a >= 0,85 (13/15)? A tarefa fica 4/4? As
arestas do calor viram válidas? As laterais ganham validade — ou
são peso morto (ou pior: a chuva calar os sensores resolve a
tarefa por desligar a entrada, e não por memória — a ablação dirá)?

Rode: python3 grafo_maior_v2.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.lif import LIF
from rede_pulsos.sinapse import Sinapse
from rede_pulsos.rede import Rede
from grafo_maior import treinar_credito, avaliar, diagnostico_inicial

PARES_MAX = 15


def montar_grafo_v2():
    rede = Rede()
    sol = rede.adicionar_neuronio(LIF("sol"))
    calor = rede.adicionar_neuronio(LIF("calor"))
    chuva = rede.adicionar_neuronio(LIF("chuva"))
    h1 = rede.adicionar_neuronio(LIF("h1"))
    h2 = rede.adicionar_neuronio(LIF("h2"))
    decisao = rede.adicionar_neuronio(LIF("decisao"))

    def lig(pre, pos, w, **kw):
        padrao = dict(stdp=True, acumular_elegibilidade=True)
        padrao.update(kw)
        rede.ligar(Sinapse(pre, pos, w, **padrao))

    # sensores -> camada oculta (calor modulador: metade do peso,
    # teto 10 — pode apoiar, nunca dirigir)
    for h in (h1, h2):
        lig(sol, h, 12.0)
        lig(calor, h, 6.0, w_max=10.0)
        lig(chuva, h, -8.0, inibitoria=True, w_min=-30.0, w_max=0.0)
    # camada oculta -> decisão
    lig(h1, decisao, 12.0)
    lig(h2, decisao, 12.0)
    # par recorrente
    lig(h1, h2, 8.0)
    lig(h2, h1, 8.0)
    # caminho direto (calor modulador também aqui: teto 5)
    lig(sol, decisao, 5.0)
    lig(calor, decisao, 3.0, w_max=5.0)
    lig(chuva, decisao, -3.0, inibitoria=True, w_min=-30.0, w_max=0.0)
    # feedback fixo: a decisão freia h1
    rede.ligar(Sinapse(decisao, h1, -5.0, stdp=False,
                       inibitoria=True, w_min=-30.0, w_max=0.0))
    # laterais entre sensores (todas inibitórias, plásticas)
    lig(chuva, sol, -6.0, inibitoria=True, w_min=-30.0, w_max=0.0)
    lig(chuva, calor, -6.0, inibitoria=True, w_min=-30.0, w_max=0.0)
    lig(sol, calor, -4.0, inibitoria=True, w_min=-30.0, w_max=0.0)
    lig(calor, sol, -4.0, inibitoria=True, w_min=-30.0, w_max=0.0)
    return rede


def pesos(rede):
    return {f"{s.pre.nome}->{s.pos.nome}": round(s.w, 2)
            for s in rede.sinapses}


def main():
    print("== Grafo maior v0.2 (6 nós, 18 sinapses, 15 pares) ==")
    rede = montar_grafo_v2()
    print(f"  início: {pesos(rede)}")
    diagnostico_inicial(rede)
    treinar_credito(rede, 1, 16)
    a16 = avaliar(rede, "16 ensaios")
    treinar_credito(rede, 17, 40)
    a40 = avaliar(rede, "40 ensaios")
    treinar_credito(rede, 41, 80)
    a80 = avaliar(rede, "80 ensaios")
    print("\n== Resumo (acertos/4 | margem | ISC_Local) ==")
    for rot, a in [("16", a16), ("40", a40), ("80", a80)]:
        print(f"  {rot} ensaios: {a[0]}/4 | {a[1]:3d} | {a[2]:.3f}")


if __name__ == "__main__":
    main()
