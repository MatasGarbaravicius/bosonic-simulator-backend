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


def cplx(z):
    return np.complex128(z[0] + 1j * z[1])


def unitary_from_gate(g):
    if g["type"] == "F":
        return PhaseShiftDescription(angle=np.float64(g["phi"]), mode_index=g["mode"])
    if g["type"] == "S":
        return SqueezingDescription(parameter=np.float64(g["z"]), mode_index=g["mode"])
    if g["type"] == "D":
        return DisplacementDescription(
            amplitude=np.array([cplx(a) for a in g["alpha"]])
        )
    raise ValueError("Unknown gate")


def build_qiskit_circuit(num_wires, term):
    qc = QuantumCircuit(num_wires)
    for g in term["gates"]:
        if g["type"] == "D":
            gate = Gate("D", num_wires, [])
            qc.append(gate, list(range(num_wires)))
        else:
            gate = Gate(g["type"], 1, [])
            qc.append(gate, [g["mode"]])
    return qc


# ---------- Routes ----------


@app.route("/render_term", methods=["POST"])
def render_term():
    data = request.json
    term_index = data["index"]
    qc = build_qiskit_circuit(data["num_wires"], data["term"])

    path = f"static/circuit_{term_index}.png"
    circuit_drawer(
        qc,
        output="mpl",
        filename=path,
        style={
            "displaycolor": {
                "F": ("#1f77b4", "#ffffff"),
                "S": ("#6a0dad", "#ffffff"),
                "D": ("#2ca02c", "#ffffff"),
            }
        },
    )
    return send_file(path, mimetype="image/png")


@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json

    superposition_terms = []
    for term in data["superposition"]:
        coeff = cplx(term["coefficient"])
        state = GaussianStateDescription.vacuum_state(data["num_wires"])
        superposition_terms.append((coeff, state))

    unitary_descriptions = []
    for term in data["superposition"]:
        for g in term["gates"]:
            unitary_descriptions.append(unitary_from_gate(g))

    amplitude = np.array(
        [cplx(a) for a in data["measurement"]["amplitude"]], dtype=np.complex128
    )

    prob = simulateexactly(
        superposition_terms,
        unitary_descriptions,
        amplitude,
        data["measurement"]["wires"],
    )

    return jsonify({"probability": float(prob)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
