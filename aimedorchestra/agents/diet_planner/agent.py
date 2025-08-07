from transformers import pipeline

class DietPlannerAgent:
    def __init__(self):
        # Initialize a text-generation model (use a lightweight model for demo)
        self.generator = pipeline('text-generation', model='gpt2', max_length=200)

    def plan(self, patient):
        """
        Generate a personalized diet plan based on patient's conditions and preferences.
        
        patient: dict with keys:
            - 'conditions': list or string of medical conditions (e.g. ['Diabetes', 'Hypertension'])
            - 'preferences': string describing dietary preferences or restrictions (optional)
        """
        conditions = patient.get('conditions', 'None')
        preferences = patient.get('preferences', 'No special preferences')

        prompt = (
            f"You are an expert dietitian. Create a 7-day personalized meal plan for a patient "
            f"with the following medical conditions: {conditions}. "
            f"Take into account these dietary preferences/restrictions: {preferences}. "
            f"Provide clear meal suggestions for each day focusing on managing their conditions."
        )
        
        # Generate diet plan using the LLM
        results = self.generator(prompt, max_length=300, num_return_sequences=1)
        diet_plan = results[0]['generated_text'][len(prompt):].strip()
        
        return diet_plan


# Example usage
if __name__ == "__main__":
    agent = DietPlannerAgent()
    patient_info = {
        'conditions': ['Type 2 Diabetes', 'Hypertension'],
        'preferences': 'Vegetarian, no nuts'
    }
    plan = agent.plan(patient_info)
    print("Diet Plan:\n", plan)
