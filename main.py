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
    BeamSplitterDescription,
)
from bosonic_simulator.algorithms.applyunitaries import applyunitaries
from bosonic_simulator.algorithms.simulateexactly import simulateexactly

app = Flask(__name__)
CORS(app)
os.makedirs("static", exist_ok=True)

# ---------- helpers ----------


def cplx(z):
    return np.complex128(z[0] + 1j * z[1])


def unitary_from_gate(g):
    if g["type"] == "F":
        return PhaseShiftDescription(angle=np.float64(g["phi"]), mode_index=g["mode"])

    if g["type"] == "S":
        return SqueezingDescription(parameter=np.float64(g["z"]), mode_index=g["mode"])

    if g["type"] == "D":
        return DisplacementDescription(
            amplitude=[complex(cplx(a)) for a in g["alpha"]
        )

    if g["type"] == "B":
        return BeamSplitterDescription(
            angle=np.float64(g["omega"]), mode_j_index=g["j"], mode_k_index=g["k"]
        )

    raise ValueError(f"Unknown gate type {g['type']}")


def build_qiskit_circuit(num_wires, gates):
    qc = QuantumCircuit(num_wires)

    for g in gates:
        if g["type"] == "D":
            gate = Gate(
                "D", num_wires, np.array([cplx(a) for a in g["alpha"]]).tolist()
            )
            qc.append(gate, list(range(num_wires)))

        elif g["type"] == "F":
            gate = Gate("F", 1, [g["phi"]])
            qc.append(gate, [g["mode"]])

        elif g["type"] == "S":
            gate = Gate("S", 1, [g["z"]])
            qc.append(gate, [g["mode"]])

        elif g["type"] == "B":
            gate = Gate("B", 2, [g["omega"]])
            qc.append(gate, [g["j"], g["k"]])

        else:
            raise ValueError(f"Unknown gate type {g['type']}")

    return qc


# ---------- rendering ----------

STYLE = {
    "displaycolor": {
        "F": ("#1f77b4", "#ffffff"),
        "S": ("#6a0dad", "#ffffff"),
        "D": ("#2ca02c", "#ffffff"),
        "B": ("#d62728", "#ffffff"),
    }
}


@app.route("/render_term", methods=["POST"])
def render_term():
    data = request.json
    idx = data["index"]
    qc = build_qiskit_circuit(data["num_wires"], data["gates"])

    path = f"static/circuit_term_{idx}.png"
    circuit_drawer(qc, output="mpl", filename=path, style=STYLE)
    return send_file(path, mimetype="image/png")


@app.route("/render_global", methods=["POST"])
def render_global():
    data = request.json
    qc = build_qiskit_circuit(data["num_wires"], data["gates"])

    path = "static/circuit_global.png"
    circuit_drawer(qc, output="mpl", filename=path, style=STYLE)
    return send_file(path, mimetype="image/png")


# ---------- simulation ----------


@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json
    n = data["num_wires"]

    # 1) prepare superposition terms
    superposition_terms = []
    for term in data["superposition"]:
        vacuum = GaussianStateDescription.vacuum_state(n)
        prepared = applyunitaries(vacuum, [unitary_from_gate(g) for g in term["gates"]])
        coeff = cplx(term["coefficient"])
        superposition_terms.append((coeff, prepared))

    # 2) global unitaries
    global_unitaries = [unitary_from_gate(g) for g in data["global_gates"]]

    # 3) measurement
    amplitude = np.array(
        [cplx(a) for a in data["measurement"]["amplitude"]],
        dtype=np.complex128,
    )

    prob = simulateexactly(
        superposition_terms,
        global_unitaries,
        amplitude,
        data["measurement"]["wires"],
    )

    return jsonify({"probability": float(prob)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
