
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# 1. ПАРАМЕТРЫ ЗАДАЧИ

a = 1.0             # Ширина ямы (безразмерный масштаб)(500)
N = 2000            # Число внутренних точек сетки
n_levels = 6        # Количество искомых состояний

# Координата xi = x/a в диапазоне [0, 1]
xi = np.linspace(0, 1, N + 2)
h = xi[1] - xi[0]
xi_inner = xi[1:-1]

# 2. ПОТЕНЦИАЛ МОРЗЕ

def potential(x):
    D = 35.0
    alpha = 5.0
    x0 = 0.32

    return D * ((1 - np.exp(-alpha * (x - x0)))**2 - 1)

U = potential(xi_inner)

# 3. ГАМИЛЬТОНИАН

# Безразмерное уравнение:
# -psi''(xi) + U(xi)*psi = E*psi

# Метод конечных разностей
main_diag = 2 / h**2 + U
off_diag = -np.ones(N - 1) / h**2

# 4. СОБСТВЕННЫЕ ЗНАЧЕНИЯ И ФУНКЦИИ

energies, eigenvectors = eigh_tridiagonal(
    main_diag,
    off_diag,
    select="i",
    select_range=(0, n_levels - 1)
)

# Нормировка волновых функций
psi = eigenvectors / np.sqrt(h)

# Добавляем нулевые значения на границах
psi_full = np.zeros((N + 2, n_levels))
psi_full[1:-1, :] = psi

print("Собственные значения энергии (E / E*):")

for i, E in enumerate(energies):
    print(f"E{i+1} = {E:.6f}")

# 5. ГРАФИК ПОТЕНЦИАЛА И УРОВНЕЙ ЭНЕРГИИ

plt.figure(figsize=(9, 6))

plt.plot(
    xi,
    potential(xi),
    color="black",
    linewidth=2,
    label="Потенциал Морзе"
)

for i, E in enumerate(energies):
    plt.axhline(
        E,
        linestyle="--",
        linewidth=1,
        label=f"E{i+1} = {E:.2f}"
    )

plt.xlabel("x / a")
plt.ylabel("Энергия / E*")
plt.title("Асимметричная яма Морзе и уровни энергии")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 6. ГРАФИКИ СОБСТВЕННЫХ ФУНКЦИЙ

fig, axes = plt.subplots(2, 3, figsize=(13, 7))

for i, ax in enumerate(axes.flat):

    ax.plot(xi, psi_full[:, i], linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8)

    ax.set_title(
        f"Состояние {i+1}, E = {energies[i]:.3f}"
    )

    ax.set_xlabel("x / a")
    ax.set_ylabel("psi")
    ax.grid(alpha=0.3)

plt.suptitle("Собственные функции уравнения Шрёдингера")
plt.tight_layout()
plt.show()

# 7. ПРОВЕРКА НОРМИРОВКИ

print("\nПроверка нормировки:")

for i in range(n_levels):
    norm = np.trapezoid(psi_full[:, i]**2, xi)
    print(f"psi{i+1}: интеграл = {norm:.6f}")
