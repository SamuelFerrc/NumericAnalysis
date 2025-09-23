import matplotlib.pyplot as plt
import math
import pandas as pd
import json

# Função para ler o arquivo de entrada
def ler_casos_arquivo(nome_arquivo):
    with open(nome_arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)
    casos = []
    for caso in dados:
        casos.append({
            "id": caso["id"],
            "f": eval(f"lambda x: {caso['f']}"),
            "df": eval(f"lambda x: {caso['df']}"),
            "x0": caso["x0"],
            "margem": caso["margem"]
        })
    return casos

# Lendo os casos
casos_teste = ler_casos_arquivo("entrada.txt")

# --- Função Newton-Raphson ---
def newton_raphson(f, df, x0, tol=1e-6, maxiter=100):
    iteracoes = []
    aproximacoes = []
    erros = []
    tabela = []

    x = x0
    for i in range(maxiter):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ValueError("Derivada nula! Método falhou.")

        x_next = x - fx / dfx
        erro = abs(fx)

        iteracoes.append(i+1)
        aproximacoes.append(x_next)
        erros.append(erro)

        tabela.append({
            "Iteração": i+1,
            "x": x,
            "f(x)": fx,
            "x_next": x_next,
            "Erro": erro
        })

        if erro < tol:
            return fx, x_next, erro, x, x_next, tol, iteracoes, aproximacoes, erros, tabela

        x = x_next

    raise ValueError("Não convergiu após o número máximo de iterações.")

# --- Execução dos casos ---
resultados = []

for caso in casos_teste:
    idx = caso["id"]
    f = caso["f"]
    df = caso["df"]
    x0 = caso["x0"]
    margem = caso["margem"]

    resultado = newton_raphson(f, df, x0, margem)
    resultados.append((idx, x0, margem, resultado))
    _, raiz, _, _, _, _, iteracoes, aprox, erros, tabela = resultado

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    axes[1].plot(iteracoes, aprox, marker="o")
    axes[1].axhline(raiz, color="red", linestyle="--", label=f"Raiz ≈ {raiz:.5f}")
    axes[1].set_title(f"Caso {idx} - Convergência da raiz")
    axes[1].set_xlabel("Iterações")
    axes[1].set_ylabel("Aproximação da raiz")
    axes[1].legend()
    axes[1].grid(True)

    axes[0].plot(iteracoes, erros, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Caso {idx} - Erro por iteração")
    axes[0].set_xlabel("Iterações")
    axes[0].set_ylabel("Erro |f(x)| (escala log)")
    axes[0].grid(True)

    plt.tight_layout()
    plt.savefig(f"grafico_caso{idx}.png")
    plt.close()

    df_tab = pd.DataFrame(tabela)
    df_tab.to_csv(f"tabela_caso{idx}.csv", index=False, float_format="%.6f", encoding="utf-8-sig")
    with open(f"tabela_caso{idx}.txt", "w", encoding="utf-8") as f_out:
        f_out.write(df_tab.to_string(index=False, float_format="%.6f"))

with open("saida.txt", "w", encoding="utf-8") as arq:
    arq.write("Resultados do Método de Newton-Raphson\n\n")
    for idx, x0, margem, res in resultados:
        arq.write(f"=== Caso {idx} ===\n")
        arq.write(f"Chute inicial: {x0}\n")
        arq.write(f"Margem de erro: {margem}\n")
        arq.write(f"f(x_final): {res[0]}\n")
        arq.write(f"Raiz aproximada: {res[1]}\n")
        arq.write(f"Erro final: {res[2]}\n")
        arq.write(f"Número de iterações: {len(res[6])}\n")
        arq.write(f"Tabela salva em tabela_caso{idx}.csv\n\n")

print("Finalizado.")
