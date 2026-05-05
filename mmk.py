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

lmbda_base = 8.0
mu = 3.0
k_list = [3,4,5,6]

Lq_arr = []
Wq_arr = []
Pwait_arr = []
for k in k_list:
    r = mmk_stats(lmbda_base, mu, k)
    Lq_arr.append(r['Lq'])
    Wq_arr.append(r['Wq'])
    Pwait_arr.append(r['P_wait'])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(k_list, Lq_arr, 'bo-', linewidth=2)
axes[0].set_xlabel('Число каналов k')
axes[0].set_ylabel('Lq')
axes[0].set_title('Средняя длина очереди')
axes[0].grid(True)

axes[1].plot(k_list, Wq_arr, 'rs-', linewidth=2)
axes[1].set_xlabel('Число каналов k')
axes[1].set_ylabel('Wq (время ожидания)')
axes[1].set_title('Среднее время ожидания')
axes[1].grid(True)

axes[2].plot(k_list, Pwait_arr, 'g^-', linewidth=2)
axes[2].set_xlabel('Число каналов k')
axes[2].set_ylabel('P_{ож}')
axes[2].set_title('Вероятность ожидания')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('metrics_vs_k.png', dpi=150)
plt.show()

mu_single = 4 * mu   # эквивалентная одноканальная система
lambda_range = np.linspace(1, 11.5, 50)
Lq_mm1 = []
Wq_mm1 = []
Lq_mm4 = []
Wq_mm4 = []

for lam in lambda_range:
    # M/M/1 с общей интенсивностью mu_single
    rho1 = lam / mu_single
    if rho1 < 1:
        Lq_mm1.append(rho1**2 / (1 - rho1))
        Wq_mm1.append(Lq_mm1[-1] / lam)
    else:
        Lq_mm1.append(np.nan)
        Wq_mm1.append(np.nan)
    # M/M/4
    try:
        res4 = mmk_stats(lam, mu, 4)
        Lq_mm4.append(res4['Lq'])
        Wq_mm4.append(res4['Wq'])
    except ValueError:
        Lq_mm4.append(np.nan)
        Wq_mm4.append(np.nan)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(lambda_range, Lq_mm1, 'r--', label='M/M/1 (общая мощность)')
ax1.plot(lambda_range, Lq_mm4, 'b-', label='M/M/4')
ax1.set_xlabel('Интенсивность поступления λ')
ax1.set_ylabel('Lq')
ax1.set_title('Длина очереди')
ax1.legend()
ax1.grid(True)

ax2.plot(lambda_range, Wq_mm1, 'r--', label='M/M/1')
ax2.plot(lambda_range, Wq_mm4, 'b-', label='M/M/4')
ax2.set_xlabel('λ')
ax2.set_ylabel('Wq')
ax2.set_title('Время ожидания')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('compare_mm1_mm4.png', dpi=150)
plt.show()

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
for ax, k_val in zip(axes3, [3,4,6]):
    r = mmk_stats(lmbda_base, mu, k_val)
    n_vals = np.arange(0, min(len(r['p']), 30))   # покажем первые 30 состояний
    probs = r['p'][:len(n_vals)]
    ax.bar(n_vals, probs, width=0.6, color='steelblue', edgecolor='black')
    ax.axvline(x=k_val, color='red', linestyle='--', label=f'k={k_val}')
    ax.set_xlabel('Число требований n')
    ax.set_ylabel('p_n')
    ax.set_title(f'Распределение при k={k_val}')
    ax.legend()
    ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig('prob_distributions.png', dpi=150)
plt.show()

t_values = np.linspace(0, 1.0, 100)
plt.figure(figsize=(8,5))
for k_val in k_list:
    r = mmk_stats(lmbda_base, mu, k_val)
    Pw = r['P_wait']
    survival = Pw * np.exp(-k_val * mu * (1 - r['rho']) * t_values)
    plt.plot(t_values, survival, label=f'k={k_val}')
plt.xlabel('Пороговое время t')
plt.ylabel('P(Wq > t)')
plt.title('Вероятность ожидания дольше t')
plt.legend()
plt.grid(True)
plt.savefig('wait_exceed.png', dpi=150)
plt.show()
