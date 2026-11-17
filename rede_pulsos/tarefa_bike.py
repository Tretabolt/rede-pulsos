"""Tarefa da bicicleta, agora com 3 sensores e 1 decisão.

Do vídeo original (neuron.machinelearning, Linear(3, 1)):
  sol 1.0 x 0.8, calor 0.7 x 0.5, chuva 0.0 x -1.5, viés -0.4
  z = +0.75 -> vai; com chuva 1.0, z = -0.75 -> fica.

Aqui cada entrada vira um neurônio sensorial, que dispara em pulsos,
e a decisão é outro LIF, que só recebe as sinapses:
  sol   -> decisão, excitatória, w_sol
  calor -> decisão, excitatória, w_calor
  chuva -> decisão, inibitória, w_chuva (negativo, sem STDP)
Corrente de cada sensor: I = 2 + 10 * valor, de 20 a 120 ms.
O "+2" não é do vídeo: é um deslocamento para o sensor não morrer.
O LIF só dispara se I > 7,5 (repouso -65 + R*I tem que cruzar -50 mV),
então I = 10 * valor deixaria o calor (0,7 -> I=7) com 0 disparos e a
rede decidiria só pelo sol. Com o deslocamento: sol I=12 (8 disparos),
calor I=9 (5), chuva I=12 (8). Como antes, não é conversão de unidades,
é tradução de ideia.
Decisão: 3 ou mais disparos da decisão em 150 ms -> "vai", senão "fica".
Essa regra de 3 é uma escolha nossa, não do vídeo — é o que separa
"disparou de verdade" de "só encostou no limiar".
"""
from .lif import LIF
from .sinapse import Sinapse
from .rede import Rede

W_SOL = 18.0
W_CALOR = 12.0
W_CHUVA = -30.0


def montar_rede(w_sol=W_SOL, w_calor=W_CALOR, w_chuva=W_CHUVA, stdp=False):
    rede = Rede()
    sol = rede.adicionar_neuronio(LIF("sol"))
    calor = rede.adicionar_neuronio(LIF("calor"))
    chuva = rede.adicionar_neuronio(LIF("chuva"))
    decisao = rede.adicionar_neuronio(LIF("decisao"))
    rede.ligar(Sinapse(sol, decisao, w_sol, stdp=stdp))
    rede.ligar(Sinapse(calor, decisao, w_calor, stdp=stdp))
    rede.ligar(Sinapse(chuva, decisao, w_chuva, stdp=False,
                       w_min=-30.0, w_max=0.0))
    return rede


def _estimulo(valor, t0=20.0, t1=120.0):
    return lambda t: (2.0 + 10.0 * valor) if t0 <= t <= t1 else 0.0


def rodar_cenario(rede, sol=1.0, calor=0.7, chuva=0.0):
    rede.resetar_ensaio(manter_pesos=True)
    out = rede.rodar({
        "sol": _estimulo(sol),
        "calor": _estimulo(calor),
        "chuva": _estimulo(chuva),
    })
    n_decisao = len(out["decisao"])
    return {
        "sol": sol, "calor": calor, "chuva": chuva,
        "disparos": out,
        "n_decisao": n_decisao,
        "vai": n_decisao >= 3,
    }


CENARIOS = [
    ("dia bom (sol, calor, sem chuva)", dict(sol=1.0, calor=0.7, chuva=0.0)),
    ("dia de chuva (sol, calor, chuva)", dict(sol=1.0, calor=0.7, chuva=1.0)),
    ("só sol", dict(sol=1.0, calor=0.0, chuva=0.0)),
    ("só calor", dict(sol=0.0, calor=0.7, chuva=0.0)),
]


def main():
    rede = montar_rede()
    for nome, cenario in CENARIOS:
        r = rodar_cenario(rede, **cenario)
        d = r["disparos"]["decisao"]
        print(f"{nome}: decisão {r['n_decisao']} disparos "
              f"{[round(x, 1) for x in d]} -> {'VAI' if r['vai'] else 'FICA'}")
    print("Pesos:", {s.pre.nome: s.w for s in rede.sinapses})


if __name__ == "__main__":
    main()
