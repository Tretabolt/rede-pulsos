"""Tarefa temporal (resposta atrasada) — a memória tem que viver no grafo.

A lição do v0.2 + escala: a tarefa da bicicleta (4 contextos, 150 ms)
não exige o grafo — um só caminho resolve — então todo mecanismo que
conserta o placar o faz simplificando a rede. Esta tarefa muda a
pergunta: em vez de "o que os sensores dizem agora", é "o que os
sensores disseram há pouco".

Ensaio de 170 ms:
  pista  20-60 ms: os sensores recebem o contexto (mesmas correntes
         da bicicleta, I = 2 + 10 * valor).
  atraso 60-170 ms: silêncio — todos os sensores com I = 0.
A decisão é medida SÓ na janela do atraso [60, 170]: 3 ou mais
disparos da decisão no atraso -> "vai", senão "fica". A mesma regra
de 3 da bicicleta, agora aplicada a uma janela em que a entrada já
morreu.

Por que um só caminho não resolve: no atraso os sensores estão
calados, então o caminho direto sensor -> decisão não carrega nada
na janela medida. A decisão só dispara no atraso se h1/h2
sustentarem atividade depois da pista — e isso só acontece pelo
par recorrente h1 <-> h2. A recorrência deixa de ser enfeite e
vira necessária por construção.

Dois detalhes honestos:
  - A elegibilidade do R-STDP aqui NÃO decai (ela acumula durante
    o ensaio e é aplicada no fim), então o intervalo de ~110 ms
    entre a pista e a recompensa é atravessável por construção
    desta implementação — não é uma propriedade da biologia, onde
    os traços decaem. Se a rede aprender, é apesar do atraso, não
    por causa de um traço que some.
  - A pista dura só 40 ms (contra 100 ms na bicicleta): os sensores
    disparam menos (sol ~3-4, calor ~2), então o laço recebe menos
    sementes. O diagnóstico inicial mostra se o grafo sem treino
    já sustenta alguma coisa no atraso.

Os 4 contextos e as respostas certas são os mesmos da bicicleta:
  dia bom (sol, calor) -> VAI; chuva -> FICA;
  só sol -> VAI; só calor -> FICA.
"""
from rede_pulsos.tarefa_bike import CENARIOS

T_MAX = 170.0
PISTA_FIM = 60.0


def _estimulo_temporal(valor, t0=20.0, t1=60.0):
    return lambda t: (2.0 + 10.0 * valor) if t0 <= t <= t1 else 0.0


def rodar_cenario_temporal(rede, sol=1.0, calor=0.7, chuva=0.0):
    rede.resetar_ensaio(manter_pesos=True)
    out = rede.rodar({
        "sol": _estimulo_temporal(sol),
        "calor": _estimulo_temporal(calor),
        "chuva": _estimulo_temporal(chuva),
    }, t_max=T_MAX)
    atraso = [t for t in out["decisao"] if t >= PISTA_FIM]
    n_atraso = len(atraso)
    return {
        "sol": sol, "calor": calor, "chuva": chuva,
        "disparos": out,
        "n_decisao": n_atraso,
        "vai": n_atraso >= 3,
    }


def main():
    from grafo_maior_v2 import montar_grafo_v2
    rede = montar_grafo_v2()
    for nome, kw in CENARIOS:
        r = rodar_cenario_temporal(rede, **kw)
        d = r["disparos"]["decisao"]
        print(f"{nome}: decisão no atraso {r['n_decisao']} disparos "
              f"{[round(x, 1) for x in d]} -> {'VAI' if r['vai'] else 'FICA'}")


if __name__ == "__main__":
    main()
