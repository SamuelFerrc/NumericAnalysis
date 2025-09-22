import matplotlib.pyplot as plt
import math
import pandas as pd
casos_teste = [
    {
        "id": 1,
        "f": lambda x: pow(x, 2) - 7,
        "x0": 2,
        "x1": 3,
        "margem": 1e-6
    },
    {
        "id": 2,
        "f": lambda x: math.exp(x) - 4 * x,
        "x0": 0,
        "x1": 1,
        "margem": 1e-6
    },
    {
        "id": 3,
        "f": lambda x: pow(x, 3) + math.cos(x),
        "x0": -1,
        "x1": 0,
        "margem": 1e-6
    }
]

def secante(f, x0, x1, tol=1e-6, maxiter=100):
    iteracoes = []
    aproximacoes = []
    erros = []
    tabela = []

    for i in range(maxiter):
        fx0 = f(x0)
        fx1 = f(x1)

        if fx1 - fx0 == 0:
            raise ZeroDivisionError("Divisão por zero no método da secante.")

        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        erro = abs(x_next - x1)

        iteracoes.append(i + 1)
        aproximacoes.append(x_next)
        erros.append(erro)

        tabela.append({
            "Iteração": i + 1,
            "x0": x0,
            "x1": x1,
            "f(x0)": fx0,
            "f(x1)": fx1,
            "x_next": x_next,
            "Erro": erro
        })

        if erro < tol:
            return f(x_next), x_next, erro, iteracoes, aproximacoes, erros, tabela

        x0, x1 = x1, x_next

    raise ValueError("Não convergiu após o número máximo de iterações.")

resultados = []
for caso in casos_teste:
    idx = caso["id"]
    f = caso["f"]
    x0 = caso["x0"]
    x1 = caso["x1"]
    margem = caso["margem"]

    resultado = secante(f, x0, x1, margem)
    resultados.append((idx, x0, x1, margem, resultado))
    fx_final, raiz, erro_final, iteracoes, aprox, erros, tabela = resultado

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
    axes[0].set_ylabel("Erro |xₙ - xₙ₋₁| (escala log)")
    axes[0].grid(True)

    plt.tight_layout()
    plt.savefig(f"grafico_caso{idx}.png")
    plt.close()

    df = pd.DataFrame(tabela)
    df.to_csv(f"tabela_caso{idx}.csv", index=False, float_format="%.6f", encoding="utf-8-sig")
    with open(f"tabela_caso{idx}.txt", "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False, float_format="%.6f"))

with open("saida.txt", "w", encoding="utf-8") as arq:
    arq.write("Resultados do Método da Secante\n\n")
    for idx, x0, x1, margem, res in resultados:
        fx_final, raiz, erro_final, iteracoes, aprox, erros, tabela = res
        arq.write(f"=== Caso {idx} ===\n")
        arq.write(f"Chute inicial x0: {x0}, x1: {x1}\n")
        arq.write(f"Margem de erro: {margem}\n")
        arq.write(f"f(x_final): {fx_final}\n")
        arq.write(f"Raiz aproximada: {raiz}\n")
        arq.write(f"Erro final: {erro_final}\n")
        arq.write(f"Número de iterações: {len(iteracoes)}\n")
        arq.write(f"Tabela salva em tabela_caso{idx}.csv\n\n")


print("Finalizado.")
