"""
Teste de invariância na rede de pulsos: a mesma quantidade,
calculada em duas representações, tem que dar o mesmo.

A quantidade é a carga sináptica total que chega no neurônio de decisão:
  Q = ∫ I_sin(t) dt, de 0 a 150 ms.

Representação A (contínua): soma I_sin(t) * dt passo a passo na simulação.
Representação B (discreta): conta os disparos do pré.
  Cada disparo do pré soma +1 em s, que decai com tau_syn = 8 ms.
  A integral de um decaimento exp(-(t-t0)/tau) de t0 até T é
  tau * (1 - exp(-(T-t0)/tau)). Pela linearidade, somando os disparos:
  Q = Σ_sinapses w * tau_syn * Σ_disparos (1 - exp(-(T-t0)/tau_syn))

O peso que converte contagem em carga é w * tau_syn — o análogo do r*
no teste cilíndrico. Esquecer o w é esquecer o jacobiano: dá um número
que não converge para o certo, não importa o refinamento.

Segunda invariância: a decisão (VAI/FICA e nº de disparos) não pode
depender do passo de integração dt. dt = 0.2, 0.1, 0.05 e 0.02 ms têm
que dar a mesma decisão nos 4 cenários da bicicleta.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from rede_pulsos.lif import LIF
from rede_pulsos.sinapse import Sinapse, TAU_SYN
from rede_pulsos.tarefa_bike import montar_rede, rodar_cenario, CENARIOS, _estimulo

T_MAX = 150.0


def rodar_com_carga(dt, sol=1.0, calor=0.7, chuva=0.0):
    """Repete a ordem de Rede.rodar, registrando I_sin na decisão."""
    rede = montar_rede(dt=dt)
    sol_n = rede.neuronios["sol"]
    calor_n = rede.neuronios["calor"]
    chuva_n = rede.neuronios["chuva"]
    decisao = rede.neuronios["decisao"]
    correntes = {
        "sol": _estimulo(sol),
        "calor": _estimulo(calor),
        "chuva": _estimulo(chuva),
    }
    n_steps = int(T_MAX / dt)
    tempos, i_sin_trace = [], []
    for i in range(n_steps):
        t = i * dt
        # I_sin no início do passo, com s e w do momento (igual a Rede)
        i_sin = sum(s.corrente() for s in rede.sinapses if s.pos.nome == "decisao")
        tempos.append(t)
        i_sin_trace.append(i_sin)
        disparou = {}
        for nome, neuronio in rede.neuronios.items():
            c = correntes.get(nome, 0.0)
            i_ext = c(t) if callable(c) else c
            i_total = i_ext + sum(s.corrente() for s in rede.sinapses if s.pos.nome == nome)
            disparou[nome] = neuronio.passo(i_total, dt, t)
        for s in rede.sinapses:
            s.apos_passo(disparou[s.pre.nome], disparou[s.pos.nome], dt, t)
    disparos = {nome: list(n.disparos) for nome, n in rede.neuronios.items()}
    # Q contínua
    q_cont = float(np.sum(i_sin_trace) * dt)
    # Q discreta, por contagem de disparos
    q_disc = 0.0
    q_sem_w = 0.0
    for s in rede.sinapses:
        for t0 in s.pre.disparos:
            fator = TAU_SYN * (1.0 - math.exp(-(T_MAX - t0) / TAU_SYN))
            q_disc += s.w * fator
            q_sem_w += 1.0 * fator
    return {
        "tempos": np.array(tempos), "i_sin": np.array(i_sin_trace),
        "disparos": disparos, "q_cont": q_cont, "q_disc": q_disc,
        "q_sem_w": q_sem_w, "n_decisao": len(disparos["decisao"]),
    }


print("=== 1. Decisão invariante ao dt ===")
print(f"{'cenário':<42} {'dt=0.2':>10} {'dt=0.1':>10} {'dt=0.05':>10} {'dt=0.02':>10}")
for nome, cenario in CENARIOS:
    linha = f"{nome:<42}"
    for dt in [0.2, 0.1, 0.05, 0.02]:
        rede = montar_rede(dt=dt)
        r = rodar_cenario(rede, **cenario)
        linha += f" {r['n_decisao']:>7} {'VAI' if r['vai'] else 'FICA':>2}"
    print(linha)

print("\n=== 2. Carga na decisão, dia bom (sol, calor, sem chuva) ===")
print(f"{'dt':>6} {'Q_cont ∫I dt':>14} {'Q_disc w·N·tau':>16} {'diferença':>10} {'Q sem w':>10}")
for dt in [0.2, 0.1, 0.05, 0.02]:
    r = rodar_com_carga(dt)
    diff = 100 * (r["q_cont"] - r["q_disc"]) / r["q_disc"]
    print(f"{dt:>6} {r['q_cont']:>14.2f} {r['q_disc']:>16.2f} {diff:>+9.1f}% {r['q_sem_w']:>10.2f}")
    if dt == 0.1:
        r01 = r

print(f"\nDia bom, dt=0.1: decisão {r01['n_decisao']} disparos, "
      f"pré sol {len(r01['disparos']['sol'])} disparos, "
      f"calor {len(r01['disparos']['calor'])} disparos.")
print(f"Q com peso = {r01['q_cont']:.2f} (contínua) vs {r01['q_disc']:.2f} (discreta).")
print(f"Q sem o w = {r01['q_sem_w']:.2f} — erra por um fator "
      f"{r01['q_cont']/r01['q_sem_w']:.2f}, como esquecer o r* no cilindro.")

# figura: I_sin(t) no dia bom, dt=0.1, e convergência de Q com dt
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Invariância na rede: contar disparos ou integrar a corrente dá o mesmo Q", fontsize=11, fontweight="bold")
ax = axes[0]
ax.plot(r01["tempos"], r01["i_sin"], color="#1a73e8", linewidth=1.2)
ax.set_xlabel("Tempo (ms)")
ax.set_ylabel("I_sin na decisão")
ax.set_title("1. Corrente sináptica, dia bom", loc="left")
ax.grid(alpha=0.25)
ax = axes[1]
dts = [0.2, 0.1, 0.05, 0.02]
q_conts = [rodar_com_carga(dt)["q_cont"] for dt in dts]
q_discs = [rodar_com_carga(dt)["q_disc"] for dt in dts]
q_sems = [rodar_com_carga(dt)["q_sem_w"] for dt in dts]
ax.plot(dts, q_conts, "o-", color="#1a73e8", label="Q contínua ∫I dt")
ax.plot(dts, q_discs, "s--", color="#34a853", label="Q discreta w·N·tau")
ax.plot(dts, q_sems, "^-", color="#e8710a", label="Q sem w (esqueceu o peso)")
ax.set_xlabel("dt (ms)")
ax.set_ylabel("Carga Q")
ax.set_title("2. Q não depende do dt; sem w, nunca chega", loc="left")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout(rect=[0, 0, 1, 0.92])
out = "/home/hatch/workspace/rede-pulsos/invariancia_rede.png"
plt.savefig(out, dpi=200)
print(f"Gráfico salvo em {out}")
