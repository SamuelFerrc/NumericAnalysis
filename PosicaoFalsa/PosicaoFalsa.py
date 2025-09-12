import matplotlib.pyplot as plt
import math
import pandas as pd

#2 - 3
#def f(x: float) -> float:
 #   return pow(x, 2) - 7

#0 - 1
def f(x: float) -> float:
    return math.exp(x) - 4 * x

#-1 - 0
#def f(x: float) -> float:
   #return pow(x, 3) + math.cos(x)



def bisseccao(inicio: float, fim: float, margem: float):
    a = float(inicio)
    b = float(fim)
    erro = float("inf")
    mid = 0.0
    encontrado = 0.0
    iteracoes = []
    mids = []
    erros = []
    tabela = []
    k = 0

    if f(a) * f(b) >= 0:
        print(f"Não tem raizes em {a} - {b}")
        return None

    while erro > margem:
        if k >= 1000:
            break
        k += 1
        mid = (a * f(b) - b * f(a))/(f(b)-f(a))
        encontrado = f(mid)
        erro = abs(encontrado)

        tabela.append({
            "k": k,
            "a": a,
            "b": b,
            "f(a)": f(a),
            "f(b)": f(b),
            "c": mid,
            "f(c)": encontrado,
            "b-a": b - a,
            "pivot": (a * f(b) - b * f(a))/(f(b)-f(a)) if a != 0 else None
        })

        iteracoes.append(k)
        mids.append(mid)
        erros.append(erro)

        if f(a) * encontrado < 0:
            b = mid
        else:
            a = mid

    return [encontrado, mid, erro, a, b, margem, iteracoes, mids, erros, tabela]


with open("entrada.txt", "r", encoding="utf-8") as arq:
    casos = [linha.strip() for linha in arq.readlines() if linha.strip()]

resultados = []

for idx, caso in enumerate(casos, start=1):
    inicio, fim, margem = map(float, caso.split())
    resultado = bisseccao(inicio, fim, margem)
    if resultado is None:
        continue

    resultados.append((idx, inicio, fim, margem, resultado))
    _, raiz, _, _, _, _, iteracoes, mids, erros, tabela = resultado

    plt.figure(figsize=(6, 4))
    plt.plot(iteracoes, mids, marker="o")
    plt.axhline(raiz, color="red", linestyle="--", label=f"Raiz ≈ {raiz:.5f}")
    plt.title(f"Caso {idx} - Convergência da raiz")
    plt.xlabel("Iterações")
    plt.ylabel("Aproximação da raiz (mid)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"grafico_raiz_caso{idx}.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(iteracoes, erros, marker="o")
    plt.yscale("log")
    plt.title(f"Caso {idx} - Erro por iteração")
    plt.xlabel("Iterações")
    plt.ylabel("Erro |f(mid)| (escala log)")
    plt.grid(True)
    plt.savefig(f"grafico_erro_caso{idx}.png")
    plt.close()

    df = pd.DataFrame(tabela)
    df.to_csv(f"tabela_caso{idx}.csv", index=False, float_format="%.6f", encoding="utf-8-sig")
    with open(f"tabela_caso{idx}.txt", "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False, float_format="%.6f"))

with open("saida.txt", "w", encoding="utf-8") as arq:
    arq.write("Resultados do Método da Bisseção\n\n")
    for idx, inicio, fim, margem, res in resultados:
        arq.write(f"=== Caso {idx} ===\n")
        arq.write(f"Início: {inicio}\n")
        arq.write(f"Fim: {fim}\n")
        arq.write(f"Margem de erro: {margem}\n")
        arq.write(f"f(mid): {res[0]}\n")
        arq.write(f"Raiz aproximada: {res[1]}\n")
        arq.write(f"Erro final: {res[2]}\n")
        arq.write(f"Intervalo final: [{res[3]}, {res[4]}]\n")
        arq.write(f"Número de iterações: {len(res[6])}\n")
        arq.write(f"Tabela salva em tabela_caso{idx}.csv\n")
        arq.write("\n")

print("Finalizado.")
