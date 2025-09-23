import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ler_casos_arquivo(nome_arquivo):
    casos = []
    with open(nome_arquivo, "r", encoding="utf-8") as f:
        linhas = f.readlines()

    caso_atual = {}
    matriz = []
    vetor_b = []
    x0 = []
    margem = 1e-6
    lendo_matriz = False
    lendo_b = False
    lendo_x0 = False

    for linha in linhas:
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            if linha.startswith("# Caso"):
                if caso_atual:
                    caso_atual["A"] = np.array(matriz, dtype=float)
                    caso_atual["b"] = np.array(vetor_b, dtype=float)
                    caso_atual["x0"] = np.array(x0, dtype=float) if x0 else np.zeros(len(vetor_b))
                    caso_atual["margem"] = margem
                    casos.append(caso_atual)
                    caso_atual = {}
                    matriz = []
                    vetor_b = []
                    x0 = []
                    margem = 1e-6
                caso_atual = {"id": int(linha.split()[-1])}
            elif linha.startswith("# Matriz A"):
                lendo_matriz = True
                lendo_b = False
                lendo_x0 = False
            elif linha.startswith("# Vetor b"):
                lendo_matriz = False
                lendo_b = True
                lendo_x0 = False
            elif linha.startswith("# x0"):
                lendo_matriz = False
                lendo_b = False
                lendo_x0 = True
            elif linha.startswith("# margem"):
                lendo_matriz = False
                lendo_b = False
                lendo_x0 = False
            continue

        if lendo_matriz:
            matriz.append([float(v) for v in linha.split()])
        elif lendo_b:
            vetor_b = [float(v) for v in linha.split()]
        elif lendo_x0:
            x0 = [float(v) for v in linha.split()]
        elif linha.startswith("# margem"):
            margem = float(linha.split()[1])

    if caso_atual:
        caso_atual["A"] = np.array(matriz, dtype=float)
        caso_atual["b"] = np.array(vetor_b, dtype=float)
        caso_atual["x0"] = np.array(x0, dtype=float) if x0 else np.zeros(len(vetor_b))
        caso_atual["margem"] = margem
        casos.append(caso_atual)

    return casos


def jacobi(A, b, x0, tol=1e-6, maxiter=100):
    n = len(b)
    x = x0.copy()
    iteracoes = []
    erros = []
    tabela = []

    for k in range(maxiter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        erro = np.linalg.norm(x_new - x, ord=np.inf)

        iteracoes.append(k + 1)
        erros.append(erro)
        tabela.append({
            "Iteração": k + 1,
            **{f"x{i + 1}": x_new[i] for i in range(n)},
            "Erro": erro
        })

        if erro < tol:
            return x_new, erro, iteracoes, erros, tabela

        x = x_new

    raise ValueError("Não convergiu após o número máximo de iterações.")


casos_teste = ler_casos_arquivo("entrada.txt")
resultados = []

for caso in casos_teste:
    idx = caso["id"]
    A = caso["A"]
    b = caso["b"]
    x0 = caso["x0"]
    margem = caso["margem"]

    raiz, erro_final, iteracoes, erros, tabela = jacobi(A, b, x0, margem)
    resultados.append((idx, x0, margem, raiz, erro_final, len(iteracoes)))

    plt.figure(figsize=(6, 4))
    plt.plot(iteracoes, erros, marker="o")
    plt.yscale("log")
    plt.title(f"Caso {idx} - Erro por iteração (Jacobi)")
    plt.xlabel("Iterações")
    plt.ylabel("Erro (||x(k+1)-x(k)||)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"grafico_caso{idx}.png")
    plt.close()

    df = pd.DataFrame(tabela)
    df.to_csv(f"tabela_caso{idx}.csv", index=False, float_format="%.6f", encoding="utf-8-sig")
    with open(f"tabela_caso{idx}.txt", "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False, float_format="%.6f"))

with open("saida.txt", "w", encoding="utf-8") as arq:
    arq.write("Resultados do Método de Jacobi\n\n")
    for idx, x0, margem, raiz, erro_final, n_iter in resultados:
        arq.write(f"=== Caso {idx} ===\n")
        arq.write(f"Chute inicial: {x0}\n")
        arq.write(f"Margem de erro: {margem}\n")
        arq.write(f"Raiz aproximada: {raiz}\n")
        arq.write(f"Erro final: {erro_final}\n")
        arq.write(f"Número de iterações: {n_iter}\n")
        arq.write(f"Tabela salva em tabela_caso{idx}.csv\n\n")

print("Finalizado.")
