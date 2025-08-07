# aimedorchestra/agents/data_synthesis/agent.py
# ─────────────────────────────────────────────────────────────────────
"""
Synthetic-data generation agent.

• Primary mode   : SDV-CTGAN  (pip install sdv)
• Fallback mode  : Faker rows (pip install faker)
"""
from __future__ import annotations

import re
from typing import Union

import numpy as np
import pandas as pd

# ─── Try modern SDV (CTGAN) ─────────────────────────────────────────
try:
    from sdv.tabular import CTGAN  # SDV ≥ 1.6
    _SDV_OK = True
except ModuleNotFoundError:
    _SDV_OK = False
    from faker import Faker
    print("⚠️  SDV not installed — using Faker-based DataSynthesizer fallback.")

# ────────────────────────────────────────────────────────────────────
# Core synthesizer class
# ────────────────────────────────────────────────────────────────────
class _DataSynthCore:
    """Internal helper that actually produces synthetic rows."""

    def __init__(self, epochs: int = 100):
        if _SDV_OK:
            self._real_seed = self._make_seed(500)
            self.model = CTGAN(epochs=epochs, verbose=False)
            self.model.fit(self._real_seed)
        else:
            self.fake = Faker()
            self.fake.seed_instance(42)

    # ---------------------------------------------------------------
    @staticmethod
    def _make_seed(n: int) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(
            {
                "age": np.random.randint(20, 80, size=n),
                "gender": np.random.choice(["Male", "Female"], size=n),
                "bmi": np.round(np.random.normal(27, 5, size=n), 1),
                "conditions": np.random.choice(
                    ["None", "Diabetes", "Hypertension", "Both"],
                    size=n,
                    p=[0.5, 0.2, 0.2, 0.1],
                ),
            }
        )

    # ---------------------------------------------------------------
    def _faker_row(self) -> dict:
        return {
            "age": np.random.randint(20, 80),
            "gender": self.fake.random_element(("Male", "Female")),
            "bmi": round(np.random.normal(27, 5), 1),
            "conditions": self.fake.random_element(("None", "Diabetes", "Hypertension", "Both")),
        }

    # ---------------------------------------------------------------
    def generate(self, n: int = 1) -> pd.DataFrame:
        if _SDV_OK:
            df = self.model.sample(n)
            df["age"] = df["age"].astype(int).clip(20, 80)
            df["bmi"] = df["bmi"].astype(float).clip(15, 45).round(1)
            return df.reset_index(drop=True)

        rows = [self._faker_row() for _ in range(n)]
        return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# Public agent class (picked up by aimedorchestra loader)
# ────────────────────────────────────────────────────────────────────
class DataSynthesisAgent:
    """
    DataSynthesisAgent
    ------------------
    run(user_input) → pd.DataFrame of synthetic patient rows.

    • user_input can be:
        "10 rows", "25", "generate 5", etc. (default = 5 rows)
    """

    def __init__(self, epochs: int = 100):
        self._core = _DataSynthCore(epochs=epochs)
        mode = "SDV-CTGAN" if _SDV_OK else "Faker"
        print(f"🧪 DataSynthesisAgent initialised ({mode} mode).")

    # ---------------------------------------------------------------
    def _parse_n(self, text: Union[str, int]) -> int:
        if isinstance(text, int):
            return max(1, text)
        m = re.search(r"\d+", text)
        return max(1, int(m.group())) if m else 5

    # ---------------------------------------------------------------
    def run(self, user_input: Union[str, int] = "") -> pd.DataFrame:
        """
        Examples
        --------
        >>> agent.run("generate 10 rows")
        >>> agent.run(3)
        """
        n = self._parse_n(user_input)
        return self._core.generate(n)


# ────────────────────────────────────────────────────────────────────
# Smoke-test
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ag = DataSynthesisAgent()
    print(ag.run("10 rows"))
