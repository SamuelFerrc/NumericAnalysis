import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def fatoracao_LU(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Pivot zero detectado.")
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    cond = np.linalg.cond(A)

    return L, U, x, cond


resultados = []

for caso in casos_teste:
    idx = caso["id"]
    A = caso["A"]
    b = caso["b"]

    L, U, x, cond = fatoracao_LU(A, b)
    resultados.append((idx, A, b, L, U, x, cond))

    df_L = pd.DataFrame(L, columns=[f"L{i+1}" for i in range(L.shape[1])])
    df_U = pd.DataFrame(U, columns=[f"U{i+1}" for i in range(U.shape[1])])
    df_completo = pd.concat([df_L, df_U], axis=1)

    df_completo.to_csv(f"caso{idx}_LU.csv", index=False, float_format="%.6f")

    with open(f"caso{idx}_LU.txt", "w", encoding="utf-8") as f:
        f.write(df_completo.to_string(index=False, float_format="%.6f"))

    plt.imshow(L, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Caso {idx} - Matriz L")
    plt.savefig(f"caso{idx}_L.png")
    plt.close()

    plt.imshow(U, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Caso {idx} - Matriz U")
    plt.savefig(f"caso{idx}_U.png")
    plt.close()

with open("saida.txt", "w") as f:
    f.write("Resultados da Fatoração LU\n\n")
    for idx, A, b, L, U, x, cond in resultados:
        f.write(f"=== Caso {idx} ===\n")
        f.write(f"Matriz A:\n{np.array(A)}\n")
        f.write(f"Vetor b: {b}\n")
        f.write(f"Solução x: {x}\n")
        f.write(f"Número de condição da matriz A: {cond:.6f}\n\n")

print("Finalizado.")
