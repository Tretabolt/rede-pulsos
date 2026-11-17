"""Memória específica: as duas correções propostas para o treino de 13/11.

Pergunta do Dan (13/11/2026): treinar dias bons e chuvosos e ver se o
peso da chuva fica inibitório enquanto sol e calor sobem. Resultado do
treino_bike.py: NÃO acontece espontaneamente — sol e calor chegam antes
do pós nos dois contextos, então o STDP fortalece o "vai" nos dois.
Variante A (chuva fixa): dia de chuva 18 VAI. Variante B (chuva com
STDP inibitório): chuva aprende até -26,41, mas dia de chuva cai só
para 8 VAI — continua errado. Falta um sinal de erro ou uma plasticidade
que fortaleça a inibição quando ela segura o pós em silêncio.

Os dois desenhos abaixo são os que a Alice descreveu ao Dan na noite de
13/11, com os mesmos parâmetros do treino original (pesos iniciais
sol 10 / calor 8 / chuva -3, A+ 0,6, A- 0,4, 16 ensaios alternados).
Nada aqui muda treino_bike.py nem a Sinapse clássica: o R-STDP usa o
modo acumular_elegibilidade da Sinapse, e a variante 2 só mexe no peso
da chuva no fim do ensaio.

Experimento 1 — R-STDP, sinal de erro (recompensa externa)
  O STDP calcula, mas não aplica na hora: acumula elegibilidade durante
  o ensaio e só no fim faz w += recompensa * elegibilidade.
  Certo = dia bom + VAI e dia de chuva + FICA -> recompensa +1.
  Errado = dia bom + FICA e dia de chuva + VAI -> recompensa -1.
  Cuidado honesto: a recompensa é dada por nós, de fora. A rede não
  descobre sozinha o que é certo — é supervisionado, não espontâneo.
  Meta: dia bom VAI, chuva FICA, só sol VAI, só calor FICA, sem saturar.

Experimento 2 — Plasticidade inibitória no silêncio
  Sobre a Variante B: se a chuva dispara e o pós fica em silêncio o
  ensaio inteiro, isso é um acerto da inibição — ela fortalece um pouco.
  No fim de cada dia de chuva, se a decisão deu 0 disparos e a chuva
  disparou 5 vezes ou mais, w_chuva -= 0,5, com teto em -30.
  A chuva ganha um pouco em todo dia de chuva; sol e calor só ganham
  quando o pós dispara — e isso vai ficando mais raro conforme a
  inibição cresce. Como a chuva não dispara no dia bom, o dia bom não
  é afetado. O 0,5 é ponto de partida para testar, não valor sagrado.
  Também é regra feita por nós — hipótese de como a biologia poderia
  fazer, não prova de que emerge sozinho.

Experimento 3 — R-STDP com atribuição de crédito (acréscimo meu)
  NÃO estava nos dois desenhos: nasceu do resultado do Experimento 1.
  A recompensa global ±1 pune também a sinapse que estava fazendo o
  certo (a chuva, que tentava inibir no dia de chuva errado). Aqui a
  inibitória recebe a recompensa com o sinal trocado: no dia de chuva
  errado, sol e calor enfraquecem (causaram o erro) e a chuva fortalece
  (tentou impedir o erro). É a mesma elegibilidade, só o crédito muda
  conforme o papel funcional da sinapse.

Rode: python3 treino_memoria_especifica.py
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.lif import LIF
from rede_pulsos.sinapse import Sinapse
from rede_pulsos.rede import Rede
from rede_pulsos.tarefa_bike import rodar_cenario
from treino_bike import W_INICIAL, teste_final

DELTA_SILENCIO = 0.5
MIN_DISPAROS_CHUVA = 5


def pesos(rede):
    return {s.pre.nome: round(s.w, 2) for s in rede.sinapses}


def montar_rstdp():
    rede = Rede()
    sol = rede.adicionar_neuronio(LIF("sol"))
    calor = rede.adicionar_neuronio(LIF("calor"))
    chuva = rede.adicionar_neuronio(LIF("chuva"))
    decisao = rede.adicionar_neuronio(LIF("decisao"))
    rede.ligar(Sinapse(sol, decisao, W_INICIAL["sol"], stdp=True,
                       acumular_elegibilidade=True))
    rede.ligar(Sinapse(calor, decisao, W_INICIAL["calor"], stdp=True,
                       acumular_elegibilidade=True))
    rede.ligar(Sinapse(chuva, decisao, W_INICIAL["chuva"], stdp=True,
                       inibitoria=True, w_min=-30.0, w_max=0.0,
                       acumular_elegibilidade=True))
    return rede


def montar_silencio():
    # Variante B do treino_bike.py: chuva com STDP inibitório.
    from treino_bike import montar
    return montar(stdp_chuva=True)


def treinar_rstdp(rede, ensaios=16, inicio=1):
    historico = []
    for i in range(inicio, inicio + ensaios):
        bom = (i % 2 == 1)  # ensaio 1, 3, 5... = dia bom (alterna como antes)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            s.aplicar_elegibilidade(recompensa)
        historico.append((i, "bom" if bom else "chuva", r["n_decisao"],
                          r["vai"], certo, recompensa, dict(pesos(rede))))
    return historico


def treinar_credito(rede, ensaios=16, inicio=1):
    """Como treinar_rstdp, mas a inibitória recebe a recompensa com o
    sinal trocado: o crédito segue o papel funcional da sinapse."""
    historico = []
    for i in range(inicio, inicio + ensaios):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            s.aplicar_elegibilidade(rec)
        historico.append((i, "bom" if bom else "chuva", r["n_decisao"],
                          r["vai"], certo, recompensa, dict(pesos(rede))))
    return historico


def treinar_silencio(rede, ensaios=16, inicio=1):
    historico = []
    sin_chuva = next(s for s in rede.sinapses if s.pre.nome == "chuva")
    for i in range(inicio, inicio + ensaios):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        reforco = ""
        if not bom and r["n_decisao"] == 0 and \
                len(r["disparos"]["chuva"]) >= MIN_DISPAROS_CHUVA:
            sin_chuva.w = max(-30.0, sin_chuva.w - DELTA_SILENCIO)
            reforco = f"silêncio: chuva {sin_chuva.w:.2f}"
        historico.append((i, "bom" if bom else "chuva", r["n_decisao"],
                          r["vai"], reforco, dict(pesos(rede))))
    return historico


def mostrar_teste(rede):
    for nome, (nd, vai) in teste_final(rede).items():
        print(f"  {nome}: {nd} {'VAI' if vai else 'FICA'}")


def main():
    print("== Experimento 1 — R-STDP (sinal de erro, recompensa ±1) ==")
    rede = montar_rstdp()
    print(f"início: {pesos(rede)}")
    for n, tipo, nd, vai, certo, rec, p in treinar_rstdp(rede):
        print(f"  ensaio {n:2d} {tipo:5s} decisão={nd:2d} "
              f"{'VAI' if vai else 'FICA'} "
              f"{'certo' if certo else 'ERRADO'} (r={rec:+d}) pesos={p}")
    print(f"final 16 ensaios: {pesos(rede)}")
    mostrar_teste(rede)
    print("  -- continua até 40 ensaios (estabilidade) --")
    for n, tipo, nd, vai, certo, rec, p in treinar_rstdp(rede, 24, inicio=17):
        if n in (24, 32, 40):
            print(f"  ensaio {n:2d} {tipo:5s} decisão={nd:2d} "
                  f"{'VAI' if vai else 'FICA'} "
                  f"{'certo' if certo else 'ERRADO'} (r={rec:+d}) pesos={p}")
    print(f"final 40 ensaios: {pesos(rede)}")
    mostrar_teste(rede)

    print("\n== Experimento 2 — Plasticidade inibitória no silêncio ==")
    rede = montar_silencio()
    print(f"início: {pesos(rede)}")
    for n, tipo, nd, vai, reforco, p in treinar_silencio(rede):
        print(f"  ensaio {n:2d} {tipo:5s} decisão={nd:2d} "
              f"{'VAI' if vai else 'FICA'} {reforco} pesos={p}")
    print(f"final 16 ensaios: {pesos(rede)}")
    mostrar_teste(rede)
    print("  -- continua até 40 ensaios (estabilidade) --")
    for n, tipo, nd, vai, reforco, p in treinar_silencio(rede, 24, inicio=17):
        if n in (24, 32, 40):
            print(f"  ensaio {n:2d} {tipo:5s} decisão={nd:2d} "
                  f"{'VAI' if vai else 'FICA'} {reforco} pesos={p}")
    print(f"final 40 ensaios: {pesos(rede)}")
    mostrar_teste(rede)

    print("\n== Experimento 3 — R-STDP com atribuição de crédito ==")
    rede = montar_rstdp()
    print(f"início: {pesos(rede)}")
    for n, tipo, nd, vai, certo, rec, p in treinar_credito(rede):
        print(f"  ensaio {n:2d} {tipo:5s} decisão={nd:2d} "
              f"{'VAI' if vai else 'FICA'} "
              f"{'certo' if certo else 'ERRADO'} (r={rec:+d}) pesos={p}")
    print(f"final 16 ensaios: {pesos(rede)}")
    mostrar_teste(rede)
    print("  -- continua até 40 ensaios (estabilidade) --")
    for n, tipo, nd, vai, certo, rec, p in treinar_credito(rede, 24, inicio=17):
        if n in (24, 32, 40):
            print(f"  ensaio {n:2d} {tipo:5s} decisão={nd:2d} "
                  f"{'VAI' if vai else 'FICA'} "
                  f"{'certo' if certo else 'ERRADO'} (r={rec:+d}) pesos={p}")
    print(f"final 40 ensaios: {pesos(rede)}")
    mostrar_teste(rede)


if __name__ == "__main__":
    main()
