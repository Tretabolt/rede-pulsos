"""Gráfico da tarefa da bicicleta: V da decisão no dia bom x no dia de chuva.

Rode: python3 plot_bike.py   (na pasta rede-pulsos)
Saída: rede_bike.png
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

import matplotlib.pyplot as plt
from rede_pulsos.rede import Rede


def rodar_com_traco(sol, calor, chuva):
    rede = Rede(registrar=True)
    # montar_rede usa Rede() sem registro; remontar aqui com registro:
    from rede_pulsos import LIF, Sinapse
    s_sol = rede.adicionar_neuronio(LIF("sol"))
    s_calor = rede.adicionar_neuronio(LIF("calor"))
    s_chuva = rede.adicionar_neuronio(LIF("chuva"))
    decisao = rede.adicionar_neuronio(LIF("decisao"))
    rede.ligar(Sinapse(s_sol, decisao, 18.0))
    rede.ligar(Sinapse(s_calor, decisao, 12.0))
    rede.ligar(Sinapse(s_chuva, decisao, -30.0, w_min=-30.0, w_max=0.0))
    from rede_pulsos.tarefa_bike import _estimulo
    out = rede.rodar({
        "sol": _estimulo(sol), "calor": _estimulo(calor), "chuva": _estimulo(chuva),
    })
    return rede.tempos, rede.traces["decisao"], out["decisao"]


t, v_bom, sp_bom = rodar_com_traco(1.0, 0.7, 0.0)
_, v_chuva, sp_chuva = rodar_com_traco(1.0, 0.7, 1.0)

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, v_bom, color="#1a73e8", linewidth=1.4,
        label=f"Dia bom: decisão dispara {len(sp_bom)}x -> VAI")
ax.plot(t, v_chuva, color="#e8710a", linewidth=1.4,
        label=f"Dia de chuva: decisão dispara {len(sp_chuva)}x -> FICA")
ax.axhline(-50, color="gray", linestyle="--", linewidth=1, label="limiar -50 mV")
ax.axvspan(20, 120, color="gray", alpha=0.07)
ax.set_xlabel("Tempo (ms)")
ax.set_ylabel("V da decisão (mV)")
ax.set_title("Rede de pulsos decide ir de bike: sol+calor disparam, chuva segura",
             loc="left")
ax.set_xlim(0, 150)
ax.set_ylim(-85, 45)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
out = "/home/hatch/workspace/rede-pulsos/rede_bike.png"
plt.savefig(out, dpi=200)
print(f"Dia bom: {len(sp_bom)} disparos {sp_bom} -> VAI")
print(f"Dia de chuva: {len(sp_chuva)} disparos -> FICA")
print(f"Gráfico salvo em {out}")
