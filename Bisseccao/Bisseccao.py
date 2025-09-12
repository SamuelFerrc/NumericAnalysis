import matplotlib.pyplot as plt

def f(x: float) -> float:
    return pow(x, 3) - 3

def bisseccao(inicio: float, fim: float, margem: float):
    i = float(inicio)
    fdir = float(fim)
    erro = float("inf")
    mid = 0.0
    encontrado = 0.0
    iteracoes = []
    mids = []
    erros = []
    k = 0
    while erro > margem:
        if k >= 1000:
            break
        k += 1
        mid = (i + fdir) / 2.0
        encontrado = f(mid)
        erro = abs(encontrado)
        iteracoes.append(k)
        mids.append(mid)
        erros.append(erro)
        if encontrado > 0:
            fdir = mid
        elif encontrado < 0:
            i = mid
        else:
            break
    return [encontrado, mid, erro, i, fdir, margem, iteracoes, mids, erros]

with open("entrada.txt", "r", encoding="utf-8") as arq:
    casos = [linha.strip() for linha in arq.readlines() if linha.strip()]

resultados = []

for idx, caso in enumerate(casos, start=1):
    inicio, fim, margem = map(float, caso.split())
    resultado = bisseccao(inicio, fim, margem)
    resultados.append((idx, inicio, fim, margem, resultado))
    _, raiz, _, _, _, _, iteracoes, mids, erros = resultado
    plt.figure(figsize=(6,4))
    plt.plot(iteracoes, mids, marker="o")
    plt.axhline(raiz, color="red", linestyle="--", label=f"Raiz ≈ {raiz:.5f}")
    plt.title(f"Caso {idx} - Convergência da raiz")
    plt.xlabel("Iterações")
    plt.ylabel("Aproximação da raiz (mid)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"grafico_raiz_caso{idx}.png")
    plt.close()
    plt.figure(figsize=(6,4))
    plt.plot(iteracoes, erros, marker="o")
    plt.yscale("log")
    plt.title(f"Caso {idx} - Erro por iteração")
    plt.xlabel("Iterações")
    plt.ylabel("Erro |f(mid)| (escala log)")
    plt.grid(True)
    plt.savefig(f"grafico_erro_caso{idx}.png")
    plt.close()

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
        arq.write("\n")

print("Finalizado.")
