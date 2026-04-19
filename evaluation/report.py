def print_section(title):
    print("\n" + "=" * 30)
    print(title)
    print("=" * 30)


def report_ground_truth(true_ate):
    print_section("GROUND TRUTH (SIMULATION ONLY)")
    print(f"True ATE (analytic): {true_ate:.6f}")


def report_model_quality(avg_uplift, corr):
    print_section("MODEL QUALITY")
    print(f"Avg model uplift: {avg_uplift:.6f}")
    print(f"Correlation with true effect: {corr:.6f}")


def report_policy_value(t_learner_value, random_value):
    print_section("POLICY PERFORMANCE")
    print(f"T-Learner policy value: {t_learner_value:.6f}")
    print(f"Random policy value: {random_value:.6f}")
    print(f"Lift: {t_learner_value - random_value:.6f}")


def report_qini(qini_auc_value):
    print_section("QINI METRICS")
    print(f"Qini AUC: {qini_auc_value:.6f}")