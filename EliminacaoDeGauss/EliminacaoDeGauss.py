import numpy as np
import pandas as pd

casos_teste = [
    {
        "id": 1,
        "A": [[ 8, -4, -2],
              [-4,  4, -2],
              [-2, -2, 10]],
        "b": [10, 0, 4]
    },
    {
        "id": 2,
        "A": [
            [ 4, -1,  0, -1,  0,  0,  0,  0,  0],
            [-1,  4, -1,  0, -1,  0,  0,  0,  0],
            [ 0, -1,  4,  0,  0, -1,  0,  0,  0],
            [-1,  0,  0,  4, -1,  0, -1,  0,  0],
            [ 0, -1,  0, -1,  4, -1,  0, -1,  0],
            [ 0,  0, -1,  0, -1,  4,  0,  0, -1],
            [ 0,  0,  0, -1,  0,  0,  4, -1,  0],
            [ 0,  0,  0,  0, -1,  0, -1,  4, -1],
            [ 0,  0,  0,  0,  0, -1,  0, -1,  4]
        ],
        "b": [0, 0, 0, 0, 0, 0, 1, 1, 1]
    },
    {
        "id": 3,
        "A": [[14,  4,  4],
              [ 4,  7, 15],
              [ 4,  7, 14]],
        "b": [100, 100, 100]
    }
]

def gauss_eliminacao(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    cond = np.linalg.cond(A)

    M = np.hstack([A, b.reshape(-1, 1)])
    tabelas = []

    for k in range(n):
        max_row = np.argmax(abs(M[k:, k])) + k
        if M[max_row, k] == 0:
            raise ValueError("Sistema sem solução única (divisão por zero).")
        if max_row != k:
            M[[k, max_row]] = M[[max_row, k]]

        for i in range(k+1, n):
            fator = M[i, k] / M[k, k]
            M[i, k:] -= fator * M[k, k:]

        tabelas.append(M.copy())

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:])) / M[i, i]

    return x, tabelas, cond

resultados = []

for caso in casos_teste:
    idx = caso["id"]
    A = caso["A"]
    b = caso["b"]

    solucao, tabelas, cond = gauss_eliminacao(A, b)
    resultados.append((idx, A, b, solucao, cond))

    with open(f"tabelas_caso{idx}.txt", "w", encoding="utf-8") as f:
        for j, tab in enumerate(tabelas):
            f.write(f"=== Etapa {j+1} ===\n")
            df = pd.DataFrame(tab)
            f.write(df.to_string(index=False, float_format="%.6f"))
            f.write("\n\n")

with open("saida.txt", "w", encoding="utf-8") as arq:
    arq.write("Resultados da Eliminação de Gauss\n\n")
    for idx, A, b, sol, cond in resultados:
        arq.write(f"=== Caso {idx} ===\n")
        arq.write(f"Matriz A:\n{np.array(A)}\n")
        arq.write(f"Vetor b: {b}\n")
        arq.write(f"Solução: {sol}\n")
        arq.write(f"Número de condição da matriz A: {cond:.6f}\n\n")

print("Finalizado.")
