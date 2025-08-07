import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
import datetime

class AuditLogger:
    def __init__(self):
        self.logs = []

    def log(self, agent_name, message):
        timestamp = datetime.datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] [{agent_name}] {message}")

    def show_logs(self):
        return "\n".join(self.logs)

class EthicsAgent:
    def __init__(self, data_synthesizer, logger=None):
        self.data_synthesizer = data_synthesizer
        self.logger = logger or AuditLogger()

    def evaluate_bias(self, n_samples=1000):
        # Generate synthetic patient data
        df = self.data_synthesizer.generate(n_samples)

        # For demo, simulate DiagnosticsAgent's binary prediction: condition positive if bmi > 30
        df['diagnosis'] = (df['bmi'] > 30).astype(int)

        # Demographic sensitive attribute: gender
        sensitive_attr = df['gender']

        # Outcome
        y_pred = df['diagnosis']

        # Compute fairness metrics using fairlearn
        metric_frame = MetricFrame(metrics=selection_rate,
                                   y_true=None,  # no ground truth here; simulating predicted selection rates
                                   y_pred=y_pred,
                                   sensitive_features=sensitive_attr)

        parity_diff = demographic_parity_difference(y_pred, sensitive_attr)

        # Log results
        self.logger.log('ETHICS', f"Selection rate by group: {metric_frame.by_group.to_dict()}")
        self.logger.log('ETHICS', f"Demographic parity difference: {parity_diff:.4f}")

        results = {
            'selection_rate_by_group': metric_frame.by_group.to_dict(),
            'demographic_parity_difference': parity_diff
        }

        return results

    def check(self, log):
        # Could implement complex logic here. For demo, just say no bias or refer to logs
        if not self.logger:
            return 'No audit logger found.'
        logs = self.logger.show_logs()
        # Simple keyword scan for warnings/errors
        if "bias" in logs.lower() or "disparity" in logs.lower():
            return 'Potential bias detected. See audit logs.'
        return 'No bias detected.'

# Example usage
if __name__ == "__main__":
    class DummySynthesizer:
        def generate(self, n=1000):
            import numpy as np
            import pandas as pd
            # Simple synthetic dataset
            np.random.seed(42)
            gender = np.random.choice(['Male', 'Female'], size=n)
            bmi = np.random.normal(27, 6, size=n)
            age = np.random.randint(20, 80, size=n)
            return pd.DataFrame({'gender': gender, 'bmi': bmi, 'age': age})

    logger = AuditLogger()
    synth = DummySynthesizer()
    ethics_agent = EthicsAgent(synth, logger)
    bias_metrics = ethics_agent.evaluate_bias()
    print("Bias metrics:", bias_metrics)
    print("Audit logs:\n", logger.show_logs())
    print("Check result:", ethics_agent.check(logger.show_logs()))
