from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import os

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.visualization import circuit_drawer

from bosonic_simulator.gaussian_state_description import GaussianStateDescription
import bosonic_simulator.gaussian_unitary_description as gud
from bosonic_simulator.algorithms.simulateexactly import simulateexactly

app = Flask(__name__)
CORS(app)
os.makedirs("static", exist_ok=True)

# ---------- Helpers ----------


def build_unitary(g):
    if g["type"] == "F":
        return gud.PhaseShiftDescription(
            angle=np.float64(g["phi"]), mode_index=g["mode"]
        )
    if g["type"] == "S":
        return gud.SqueezingDescription(
            parameter=np.float64(g["z"]), mode_index=g["mode"]
        )

    if g["type"] == "B":
        return gud.BeamSplitterDescription(np.float64(0.1), 0, 1)

    if g["type"] == "D":
        return gud.DisplacementDescription(np.zeros(1, dtype=complex))

    return gud.DisplacementDescription(np.zeros(1, dtype=complex))


def build_qiskit_circuit(data):
    qc = QuantumCircuit(data["num_wires"])
    for g in data["gates"]:
        gate = Gate(g["type"], 1, [])
        qc.append(gate, [g["mode"]])
    return qc


# ---------- Routes ----------


@app.route("/render", methods=["POST"])
def render():
    qc = build_qiskit_circuit(request.json)
    path = "static/circuit.png"
    circuit_drawer(
        qc,
        output="mpl",
        filename=path,
        style={
            "displaycolor": {
                "F": ("#1f77b4", "#fff"),
                "S": ("#6a0dad", "#fff"),
            }
        },
    )
    return send_file(path, mimetype="image/png")


@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json

    state = GaussianStateDescription.vacuum_state(data["num_wires"])
    superposition = [(np.complex128(1 + 0j), state)]

    unitaries = [build_unitary(g) for g in data["gates"]]

    amplitude = np.array(
        [complex(a[0], a[1]) for a in data["measurement"]["amplitude"]],
        dtype=np.complex128,
    )

    prob = simulateexactly(
        superposition, unitaries, amplitude, data["measurement"]["wires"]
    )

    return jsonify({"probability": float(prob)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
