# treatment_agent.py  ───────────────────────────────────────────────
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# ───────────────────────────────────────────────────────────────────
# 1. Environment
# ───────────────────────────────────────────────────────────────────
class TreatmentEnv(gym.Env):
    """
    State  : [severity]   ∈  [0,10]
    Action : 0-3 dose level
    Reward : 10 - severity  (higher reward = healthier)
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.action_space      = gym.spaces.Discrete(4)

        self.max_steps = 20
        self.reset()

    # Gym API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state      = np.array([5.0], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        dose_effect = action * np.random.uniform(0.5, 1.5)
        noise       = np.random.normal(0, 0.5)
        self.state  = np.clip(self.state - dose_effect + noise, 0, 10)
        self.step_count += 1

        reward = 10 - self.state[0]
        done   = self.step_count >= self.max_steps or self.state[0] <= 0.1
        return self.state, reward, done, False, {}

    def render(self):
        print(f"Step {self.step_count}: Severity = {self.state[0]:.2f}")

# ───────────────────────────────────────────────────────────────────
# 2. Agent
# ───────────────────────────────────────────────────────────────────
class TreatmentAgent:
    """
    PPO policy trained in a toy patient-response simulator.
    plan(patient) accepts:
        • float  -> treated as severity directly
        • dict   -> {'age': …, 'conditions': […] , 'severity': … (optional)}
    Returns: human-friendly dose string.
    """
    _DOSE_MAP = {0: "No dose", 1: "Low dose", 2: "Medium dose", 3: "High dose"}

    def __init__(self, timesteps: int = 10_000):
        self.env   = make_vec_env(TreatmentEnv, n_envs=1)
        self.model = PPO("MlpPolicy", self.env, verbose=0)
        self.model.learn(total_timesteps=timesteps)

    # --------------------------------------------------------------
    def plan(self, patient):
        """Choose best dose for a patient (float severity *or* patient dict)."""
        severity = self._severity_from_patient(patient)
        obs      = np.array([[severity]], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return self._DOSE_MAP.get(int(action[0]), "Unknown")

    # --------------------------------------------------------------
    @staticmethod
    def _severity_from_patient(patient):
        """Heuristic → convert patient dict to 0-10 severity score."""
        # Case 1: already a number
        if isinstance(patient, (int, float, np.number)):
            return float(patient)

        # Case 2: dict with optional explicit severity
        if isinstance(patient, dict):
            if "severity" in patient:
                return float(patient["severity"])

            # Toy heuristics (replace with real model / rules as needed)
            base = 5.0
            age_factor = max(0, (patient.get("age", 40) - 40) * 0.05)    # +0.05 per yr > 40
            cond_factor = 1.0 * len(patient.get("conditions", []))        # +1 per condition
            sev = np.clip(base + age_factor + cond_factor, 0, 10)
            return float(sev)

        raise TypeError("patient must be a float or a dict")

# ───────────────────────────────────────────────────────────────────
# Quick test
if __name__ == "__main__":
    print("Training TreatmentAgent...")
    agent = TreatmentAgent(timesteps=5_000)

    # Dict input
    patient_dict = {"age": 50, "conditions": ["Hypertension"]}
    print("Plan for", patient_dict, "→", agent.plan(patient_dict))

    # Direct severity
    print("Plan for severity 7.0 →", agent.plan(7.0))
