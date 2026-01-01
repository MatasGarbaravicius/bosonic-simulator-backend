from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import circuit_drawer
import os

app = Flask(__name__)
CORS(app)

os.makedirs("static", exist_ok=True)


def build_circuit(data):
    qc = QuantumCircuit(data["wires"], data["wires"])
    for g in data["gates"]:
        if g["type"] == "RX":
            qc.rx(g["angle"], g["wire"])
        elif g["type"] == "RY":
            qc.ry(g["angle"], g["wire"])
        elif g["type"] == "RZ":
            qc.rz(g["angle"], g["wire"])
        elif g["type"] == "MEASURE":
            qc.measure(g["wire"], g["wire"])
    return qc


@app.route("/render", methods=["POST"])
def render():
    qc = build_circuit(request.json)
    path = "static/circuit.png"
    circuit_drawer(qc, output="mpl", filename=path)
    return send_file(path, mimetype="image/png")


@app.route("/simulate", methods=["POST"])
def simulate():
    qc = build_circuit(request.json)
    if qc.num_clbits == 0:
        return jsonify({"error": "No measurements in circuit"})
    sim = Aer.get_backend("qasm_simulator")
    tqc = transpile(qc, sim)
    job = sim.run(tqc, shots=1024)
    return jsonify({"counts": job.result().get_counts()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
