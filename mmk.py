import math
import numpy as np
import matplotlib.pyplot as plt

def mmk_stats(lmbda, mu, k, max_n=200):
    """
    Расчёт стационарных характеристик системы M/M/k.
    
    Параметры:
        lmbda  -- интенсивность входящего потока
        mu     -- интенсивность обслуживания одного прибора
        k      -- число приборов
        max_n  -- максимальное учитываемое число состояний
    Возвращает словарь с rho, p0, p, Lq, L, Wq, W, P_wait.
    """
    rho = lmbda / (k * mu)
    if rho >= 1:
        raise ValueError("Система нестационарна (rho >= 1)")

    # Вычисление p0
    sum_p0 = sum((k * rho)**n / math.factorial(n) for n in range(k))
    sum_p0 += (k * rho)**k / (math.factorial(k) * (1 - rho))
    p0 = 1.0 / sum_p0

    # Стационарные вероятности
    p = [0.0] * (max_n + 1)
    p[0] = p0
    for n in range(1, k + 1):
        p[n] = (k * rho)**n / math.factorial(n) * p0
    for n in range(k + 1, max_n + 1):
        p[n] = (k**k * rho**n) / math.factorial(k) * p0

    # Характеристики
    Lq = sum((n - k) * p[n] for n in range(k + 1, max_n + 1))
    L = Lq + k * rho
    Wq = Lq / lmbda
    W = L / lmbda
    P_wait = sum(p[n] for n in range(k, max_n + 1))

    return {
        'rho': rho,
        'p0': p0,
        'p': p,
        'Lq': Lq,
        'L': L,
        'Wq': Wq,
        'W': W,
        'P_wait': P_wait
    }


# Численные эксперименты

if __name__ == "__main__":
    lmbda = 8.0
    mu = 3.0
    k_values = [3, 4, 5, 6]

    print(f"{'k':<4} {'rho':<8} {'p0':<8} {'Lq':<8} {'L':<8} {'Wq':<8} {'P_wait':<8}")
    print("-" * 58)
    for k in k_values:
        res = mmk_stats(lmbda, mu, k)
        print(f"{k:<4} {res['rho']:<8.4f} {res['p0']:<8.4f} "
              f"{res['Lq']:<8.4f} {res['L']:<8.4f} "
              f"{res['Wq']:<8.4f} {res['P_wait']:<8.4f}")

    # График Lq в зависимости от rho для разных k
    k_fixed = 4
    rho_range = np.linspace(0.1, 0.95, 50)
    Lq_vals = [mmk_stats(r * k_fixed * mu, mu, k_fixed)['Lq'] for r in rho_range]

    plt.figure(figsize=(8, 5))
    plt.plot(rho_range, Lq_vals, 'b-', linewidth=2)
    plt.xlabel('Коэффициент нагрузки ρ')
    plt.ylabel('Среднее число требований в очереди Lq')
    plt.title(f'Зависимость Lq от ρ для системы M/M/{k_fixed}')
    plt.grid(True)
    plt.savefig('lq_vs_rho.png', dpi=150)
    plt.show()