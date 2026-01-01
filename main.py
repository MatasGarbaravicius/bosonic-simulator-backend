from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import os

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.visualization import circuit_drawer

from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.gaussian_unitary_description import (
    PhaseShiftDescription,
    SqueezingDescription,
    DisplacementDescription,
)
from bosonic_simulator.algorithms.simulateexactly import simulateexactly

app = Flask(__name__)
CORS(app)
os.makedirs("static", exist_ok=True)

# ---------- Helpers ----------


def cplx(x):
    return np.complex128(x[0] + 1j * x[1])


def build_unitary(g):
    if g["type"] == "F":
        return PhaseShiftDescription(angle=np.float64(g["phi"]), mode_index=g["mode"])
    if g["type"] == "S":
        return SqueezingDescription(parameter=np.float64(g["z"]), mode_index=g["mode"])
    if g["type"] == "D":
        return DisplacementDescription(
            amplitude=np.array([cplx(a) for a in g["alpha"]])
        )


def build_qiskit_circuit(data):
    qc = QuantumCircuit(data["num_wires"])
    for term in data["superposition"]:
        for g in term["gates"]:
            if g["type"] == "D":
                gate = Gate("D", data["num_wires"], [])
                qc.append(gate, list(range(data["num_wires"])))
            else:
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
                "D": ("#2ca02c", "#fff"),
            }
        },
    )
    return send_file(path, mimetype="image/png")


@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json

    superposition = []
    for term in data["superposition"]:
        state = GaussianStateDescription.vacuum_state(data["num_wires"])
        coeff = cplx(term["coefficient"])
        superposition.append((coeff, state))

    unitaries = []
    for term in data["superposition"]:
        for g in term["gates"]:
            unitaries.append(build_unitary(g))

    amplitude = np.array(
        [cplx(a) for a in data["measurement"]["amplitude"]], dtype=np.complex128
    )

    prob = simulateexactly(
        superposition, unitaries, amplitude, data["measurement"]["wires"]
    )

    return jsonify({"probability": float(prob)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
