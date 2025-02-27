import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.utils
from scipy import stats
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from statsmodels.stats.power import TTestIndPower
from scipy.stats import permutation_test

sns.set(style="whitegrid")
np.random.seed(42)


def compute_bootstrap_means(data, bootstrap_iterations=10000, sample_size=50):
    bootstrap_means = []
    for _ in range(bootstrap_iterations):
        boot_sample = sklearn.utils.resample(data, n_samples=sample_size, replace=True)
        bootstrap_means.append(np.mean(boot_sample))
    return np.array(bootstrap_means)


def draw_histogram(data, title, x_label, y_label, color='skyblue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=50, kde=True, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def check_normality(data, alpha=0.05):
    _, p_value = stats.normaltest(data)
    print(f"Test normalności  – p-value: {p_value:.4f}")
    if p_value < alpha:
        print("Odrzucamy H0: rozkład nie jest normalny\n")
        return False
    else:
        print("Brak podstaw do odrzucenia H0: rozkład jest normalny\n")
        return True


def get_percentile_ci(data, alpha=0.05):
    lower = 100 * (alpha / 2)
    upper = 100 * (1 - alpha / 2)
    return np.percentile(data, [lower, upper])


def get_t_ci(data, alpha=0.05):
    mean_val = np.mean(data)
    se_val = np.std(data, ddof=1) / np.sqrt(len(data))
    return stats.t.interval(1 - alpha, df=len(data) - 1, loc=mean_val, scale=se_val)


def get_bootstrap_ci(data, stat_func=bs_stats.mean, alpha=0.05, bootstrap_iterations=10000):
    return bs.bootstrap(data, stat_func=stat_func, alpha=alpha, num_iterations=bootstrap_iterations)


def print_bootstrap_result(boot_result, description="parametru"):
    print(f"Przedział ufności (bootstrap) dla {description}:")
    print(f"   Wartość oszacowania: {boot_result.value:.4f}")
    print(f"   Dolna granica:       {boot_result.lower_bound:.4f}")
    print(f"   Górna granica:       {boot_result.upper_bound:.4f}\n")


def cohen_d(group_x, group_y):
    nx = len(group_x)
    ny = len(group_y)
    dof = nx + ny - 2
    mean_x, mean_y = np.mean(group_x), np.mean(group_y)
    var_x, var_y = np.var(group_x, ddof=1), np.var(group_y, ddof=1)
    spooled = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / dof)
    return (mean_x - mean_y) / spooled


def analyze_normal_distribution():
    print(" ROZKŁAD NORMALNY ")
    x1_mean = 0
    x1_stddev = 1
    n_samples = 50
    data = np.random.normal(loc=x1_mean, scale=x1_stddev, size=n_samples)
    n_boot = 10000
    bootstrap_means_list = []

    for _ in range(n_boot):
        boot_sample = sklearn.utils.resample(data, replace=True, n_samples=n_samples)
        bootstrap_means_list.append(np.mean(boot_sample))
    bootstrap_means_list = np.array(bootstrap_means_list)

    print("\nHistogram średnich z próbkowania bootstrapowego (rozkład normalny):")
    draw_histogram(bootstrap_means_list, "Średnie bootstrapowe (Normalny)", "Średnia", "Częstotliwość")

    print("Test normalności dla średnich bootstrapowych:")
    check_normality(bootstrap_means_list)

    mean_boot = np.mean(bootstrap_means_list)
    std_boot = np.std(bootstrap_means_list)
    print(f"Średnia średnich bootstrapowych: {mean_boot:.4f}")
    print(f"Odchylenie std. średnich bootstrapowych: {std_boot:.4f}\n")

    alpha = 0.05
    ci_lower_perc, ci_upper_perc = get_percentile_ci(bootstrap_means_list, alpha=alpha)
    print(f"Przedział ufności (95%) – metoda percentylowa: ({ci_lower_perc:.4f}, {ci_upper_perc:.4f})")

    ci_t = get_t_ci(data, alpha=alpha)
    print(f"Przedział ufności (95%) – metoda t-Studenta:   ({ci_t[0]:.4f}, {ci_t[1]:.4f})")

    bs_mean_int = get_bootstrap_ci(data, stat_func=bs_stats.mean, alpha=alpha, bootstrap_iterations=n_boot)
    bs_std_int = get_bootstrap_ci(data, stat_func=bs_stats.std, alpha=alpha, bootstrap_iterations=n_boot)
    print_bootstrap_result(bs_mean_int, description="średniej")
    print_bootstrap_result(bs_std_int, description="odchylenia standardowego")

    se_classic = np.std(data, ddof=1) / np.sqrt(len(data))
    std_boot_means = np.std(bootstrap_means_list, ddof=1)
    print("Porównanie klasycznego błędu standardowego (SE) i odchylenia średnich bootstrapowych:")
    print(f"   Klasyczny SE: {se_classic:.4f}")
    print(f"   Odch. std. z średnich bootstrap: {std_boot_means:.4f}\n")


def analyze_skew_distribution():
    print("ROZKŁAD SKOŚNY")
    skew_data = stats.skewnorm.rvs(a=5, size=50)
    print("\nHistogram rozkładu skośnego (Skewnorm):")
    draw_histogram(skew_data, "Histogram – Skewnorm", "Wartość", "Częstotliwość", color="orange")
    print(f"Średnia (empiryczna): {np.mean(skew_data):.4f}")
    print(f"Mediana (empiryczna): {np.median(skew_data):.4f}\n")

    n_samples = len(skew_data)
    n_boot = 10000
    skew_boot_means = []
    for _ in range(n_boot):
        boot_sample = sklearn.utils.resample(skew_data, replace=True, n_samples=n_samples)
        skew_boot_means.append(np.mean(boot_sample))
    skew_boot_means = np.array(skew_boot_means)

    alpha = 0.05
    ci_lower_perc, ci_upper_perc = get_percentile_ci(skew_boot_means, alpha=alpha)
    print(f"Przedział ufności (95%) – metoda percentylowa: ({ci_lower_perc:.4f}, {ci_upper_perc:.4f})")

    ci_t = get_t_ci(skew_data, alpha=alpha)
    print(f"Przedział ufności (95%) – metoda t-Studenta:   ({ci_t[0]:.4f}, {ci_t[1]:.4f})\n")

    bs_mean_int_skew = get_bootstrap_ci(skew_data, stat_func=bs_stats.mean, alpha=alpha, bootstrap_iterations=n_boot)
    print_bootstrap_result(bs_mean_int_skew, description="średniej (rozkład skośny)")


def perform_ab_testing():
    print(" TESTY A/B")
    n_samples = 50
    group_a = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    group_b = np.random.normal(loc=0.2, scale=0.5, size=n_samples)

    a_boot_means = compute_bootstrap_means(group_a, bootstrap_iterations=10000, sample_size=n_samples)
    ci_lower_perc, ci_upper_perc = get_percentile_ci(a_boot_means, alpha=0.05)

    print(f"\nŚrednia (empiryczna) A: {np.mean(group_a):.4f}")
    print(f"Przedział ufności (95%) A (met. percentylowa): ({ci_lower_perc:.4f}, {ci_upper_perc:.4f})")
    print(f"Czy średnia B = {np.mean(group_b):.4f} mieści się w przedziale ufności A?")
    if (np.mean(group_b) >= ci_lower_perc) and (np.mean(group_b) <= ci_upper_perc):
        print(" -> Nie odrzucamy H0 (jednokierunkowo)")
    else:
        print(" -> Odrzucamy H0 (jednokierunkowo)")

    t_stat, p_value_t = stats.ttest_ind(group_a, group_b, equal_var=False)
    print(f"\nTest t (dwukierunkowy): stat={t_stat:.4f}, p-value={p_value_t:.4f}")
    alpha = 0.05
    if p_value_t < alpha:
        print(" Odrzucamy H0 (Test t). Różnica istotna statystycznie.\n")
    else:
        print(" Brak podstaw do odrzucenia H0 (Test t).\n")

    def mean_diff_statistic(x, y, axis=0):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    perm_result = permutation_test((group_a, group_b), mean_diff_statistic, n_resamples=1000, alternative='two-sided')
    p_value_perm = perm_result.pvalue
    print(f"Test permutacyjny p-value={p_value_perm:.4f}")
    if p_value_perm < alpha:
        print(" Odrzucamy H0.\n")
    else:
        print(" brak podstaw do odrzucenia H0.\n")


def estimate_test_power():
    print(" MOC TESTU ")
    n_samples = 50
    alpha = 0.05

    sample_1 = np.random.normal(loc=0, scale=1, size=n_samples)
    sample_2 = np.random.normal(loc=0.2, scale=0.5, size=n_samples)

    iterations = 1000
    count_significant = 0
    for _ in range(iterations):
        x1 = np.random.normal(loc=0, scale=1, size=n_samples)
        x2 = np.random.normal(loc=0.2, scale=0.5, size=n_samples)
        _, p_val = stats.ttest_ind(x1, x2, equal_var=False)
        if p_val < alpha:
            count_significant += 1

    empirical_power = count_significant / iterations
    print(f"Empirycznie oszacowana moc testu na podstawie {iterations} powtórzeń: {empirical_power:.3f}")

    effect_size = cohen_d(sample_1, sample_2)
    print(f"Wielkość efektu : {effect_size:.4f}")

    analysis = TTestIndPower()
    power_val = analysis.solve_power(effect_size=effect_size, nobs1=n_samples, alpha=alpha, ratio=1.0, alternative='two-sided')
    print(f"Moc testu przy n={n_samples}, alpha={alpha}, effect_size={effect_size:.4f}: {power_val:.3f}")

    desired_power = 0.90
    sample_size_needed = analysis.solve_power(effect_size=effect_size, power=desired_power, alpha=alpha, ratio=1.0)
    print(f"Aby uzyskać moc testu 90%, potrzebne jest ~{sample_size_needed:.1f} obserwacji na grupę.")


def compare_vaccines():
    print(" SZCZEPIONKI (PFIZER VS. MODERNA) ")

    pfizer_n = 21500
    pfizer_infected = 9
    pfizer_data = np.array([0] * pfizer_infected + [1] * (pfizer_n - pfizer_infected))

    moderna_n = 15000
    moderna_infected = 5
    moderna_data = np.array([0] * moderna_infected + [1] * (moderna_n - moderna_infected))

    pfizer_eff = np.mean(pfizer_data)
    moderna_eff = np.mean(moderna_data)

    print(f"\nPfizer – średnia skuteczność: {pfizer_eff:.6f}")
    print(f"Moderna – średnia skuteczność: {moderna_eff:.6f}")

    pfizer_std = np.std(pfizer_data, ddof=1)
    moderna_std = np.std(moderna_data, ddof=1)
    print(f"Pfizer – odchylenie std: {pfizer_std:.6f}")
    print(f"Moderna – odchylenie std: {moderna_std:.6f}")

    pfizer_ci = get_bootstrap_ci(pfizer_data, stat_func=bs_stats.mean, alpha=0.05, bootstrap_iterations=10000)
    moderna_ci = get_bootstrap_ci(moderna_data, stat_func=bs_stats.mean, alpha=0.05, bootstrap_iterations=10000)

    print_bootstrap_result(pfizer_ci, description="skuteczności (Pfizer)")
    print_bootstrap_result(moderna_ci, description="skuteczności (Moderna)")

    t_stat, p_val = stats.ttest_ind(pfizer_data, moderna_data, equal_var=False)
    alpha = 0.05
    print(f"Test t  p-value = {p_val:.8f}")
    if p_val < alpha:
        print("Różnica istotna statystycznie (Test t)\n")
    else:
        print(" Brak podstaw do odrzucenia H0 (Test t)\n")

    def mean_diff_statistic(x, y, axis=0):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    perm_res = permutation_test((pfizer_data, moderna_data), mean_diff_statistic, n_resamples=10000, alternative='two-sided')
    p_val_perm = perm_res.pvalue
    print(f"Test permutacyjny , p-value = {p_val_perm:.8f}")
    if p_val_perm < alpha:
        print("Różnica istotna statystycznie (Test permutacyjny)")
    else:
        print("Brak podstaw do odrzucenia H0 (Test permutacyjny)")
    print()

    effect_size = cohen_d(pfizer_data, moderna_data)
    analysis = TTestIndPower()
    current_power = analysis.solve_power(effect_size=effect_size,
                                         nobs1=len(pfizer_data),
                                         alpha=alpha,
                                         ratio=len(moderna_data)/len(pfizer_data))
    print(f"Oszacowana moc testu: {current_power:.3f}")

    desired_power = 0.60
    sample_size_needed = analysis.solve_power(effect_size=effect_size,
                                              power=desired_power,
                                              alpha=alpha,
                                              ratio=len(moderna_data)/len(pfizer_data))
    print(f"Liczba obserwacji (Pfizer), aby uzyskać moc 60%: {sample_size_needed:.0f}")
    print("Oba przedziały ufności częściowo się pokrywają, co sugeruje brak wyraźnej przewagi jednej szczepionki nad drugą.")
    print(" Zarówno test t, jak i test permutacyjny wykazały brak statystycznie istotnej różnicy między skutecznościami szczepionek Pfizer i Moderna")
if __name__ == "__main__":
    analyze_normal_distribution()
    analyze_skew_distribution()
    perform_ab_testing()
    estimate_test_power()
    compare_vaccines()




