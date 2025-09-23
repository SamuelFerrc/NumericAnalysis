import matplotlib.pyplot as plt
import math
import pandas as pd
import json

def ler_casos_arquivo(nome_arquivo):
    with open(nome_arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)
    casos = []
    for caso in dados:
        casos.append({
            "id": caso["id"],
            "f": eval(f"lambda x: {caso['f']}"),
            "inicio": caso["inicio"],
            "fim": caso["fim"],
            "margem": caso["margem"]
        })
    return casos

casos_teste = ler_casos_arquivo("entrada.txt")

def PosicaoFalsa(f, inicio: float, fim: float, margem: float):
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
        print(f"Não tem raízes em {a} - {b}")
        return None

    while erro > margem:
        if k >= 1000:
            break
        k += 1
        mid = (a * f(b) - b * f(a)) / (f(b) - f(a))
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
            "pivot": (a * f(b) - b * f(a)) / (f(b) - f(a)) if a != 0 else None
        })

        iteracoes.append(k)
        mids.append(mid)
        erros.append(erro)

        if f(a) * encontrado < 0:
            b = mid
        else:
            a = mid

    return [encontrado, mid, erro, a, b, margem, iteracoes, mids, erros, tabela]

resultados = []

for caso in casos_teste:
    idx = caso["id"]
    f = caso["f"]
    inicio = caso["inicio"]
    fim = caso["fim"]
    margem = caso["margem"]

    resultado = PosicaoFalsa(f, inicio, fim, margem)
    if resultado is None:
        continue

    resultados.append((idx, inicio, fim, margem, resultado))
    _, raiz, _, _, _, _, iteracoes, mids, erros, tabela = resultado

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    axes[1].plot(iteracoes, mids, marker="o")
    axes[1].axhline(raiz, color="red", linestyle="--", label=f"Raiz ≈ {raiz:.5f}")
    axes[1].set_title(f"Caso {idx} - Convergência da raiz")
    axes[1].set_xlabel("Iterações")
    axes[1].set_ylabel("Aproximação da raiz (mid)")
    axes[1].legend()
    axes[1].grid(True)

    axes[0].plot(iteracoes, erros, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Caso {idx} - Erro por iteração")
    axes[0].set_xlabel("Iterações")
    axes[0].set_ylabel("Erro |f(mid)| (escala log)")
    axes[0].grid(True)

    plt.tight_layout()
    plt.savefig(f"grafico_caso{idx}.png")
    plt.close()

    df = pd.DataFrame(tabela)
    df.to_csv(f"tabela_caso{idx}.csv", index=False, float_format="%.6f", encoding="utf-8-sig")
    with open(f"tabela_caso{idx}.txt", "w", encoding="utf-8") as f_out:
        f_out.write(df.to_string(index=False, float_format="%.6f"))

with open("saida.txt", "w", encoding="utf-8") as arq:
    arq.write("Resultados do Método da Posição Falsa\n\n")
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
        arq.write(f"Tabela salva em tabela_caso{idx}.csv\n\n")

print("Finalizado.")
