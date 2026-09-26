"""
generate_data.py
-----------------
Generates a synthetic dataset for the Investment Risk Profiling project.

Each row represents an investor described by demographic and financial
attributes. A rule-based formula (with added noise) assigns each investor
to a risk tolerance category: Conservative, Moderate, or Aggressive.

Run this once to produce investor_data.csv, which train_model.py consumes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N_SAMPLES = 3000


def generate_dataset(n=N_SAMPLES):
    age = np.random.randint(18, 71, n)
    annual_income = np.random.randint(20_000, 200_001, n)
    investment_amount = np.random.randint(1_000, 100_001, n)
    years_experience = np.random.randint(0, 31, n)
    financial_knowledge = np.random.randint(1, 11, n)          # 1-10 scale
    portfolio_diversity = np.random.randint(1, 11, n)          # 1-10 scale
    debt_to_income = np.round(np.random.uniform(0, 1, n), 2)   # 0-1 ratio

    # Normalize each factor to a 0-1 scale so they can be combined fairly
    age_score = 1 - (age - 18) / (70 - 18)                      # younger -> higher risk appetite
    income_score = (annual_income - 20_000) / (200_000 - 20_000)
    invest_score = (investment_amount - 1_000) / (100_000 - 1_000)
    exp_score = years_experience / 30
    knowledge_score = (financial_knowledge - 1) / 9
    diversity_score = (portfolio_diversity - 1) / 9
    debt_score = 1 - debt_to_income                            # lower debt -> higher risk appetite

    # Weighted combination -> overall "risk appetite" score (0-1)
    raw_score = (
        0.20 * age_score
        + 0.15 * income_score
        + 0.15 * invest_score
        + 0.15 * exp_score
        + 0.15 * knowledge_score
        + 0.10 * diversity_score
        + 0.10 * debt_score
    )

    # Add noise so the problem isn't trivially separable
    noise = np.random.normal(0, 0.045, n)
    final_score = np.clip(raw_score + noise, 0, 1)

    risk_category = pd.cut(
        final_score,
        bins=[-0.01, 0.38, 0.68, 1.01],
        labels=["Conservative", "Moderate", "Aggressive"],
    )

    df = pd.DataFrame(
        {
            "Age": age,
            "Annual_Income": annual_income,
            "Investment_Amount": investment_amount,
            "Years_Experience": years_experience,
            "Financial_Knowledge_Score": financial_knowledge,
            "Portfolio_Diversity_Score": portfolio_diversity,
            "Debt_to_Income_Ratio": debt_to_income,
            "Risk_Category": risk_category.astype(str),
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("investor_data.csv", index=False)
    print(f"Saved {len(df)} rows to investor_data.csv")
    print(df["Risk_Category"].value_counts())
