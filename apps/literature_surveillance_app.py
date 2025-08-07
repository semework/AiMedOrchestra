from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.getcwd())
from aimedorchestra.agents.literature_surveillance.agent import LiteratureAgent

app = Flask(__name__)
agent = LiteratureAgent()

@app.route('/run', methods=['POST'])
def run_agent():
    data = request.json
    input_data = data.get('input', None)
    if input_data is None:
        return jsonify({"error": "Missing 'input' field in JSON"}), 400
    try:
        result = agent.search(input_data)
        return jsonify({"result": str(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)