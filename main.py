from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.visualization import circuit_drawer
import numpy as np
import os, copy, json, base64

from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.gaussian_unitary_description import (
    DisplacementDescription,
    PhaseShiftDescription,
    SqueezingDescription,
    BeamSplitterDescription,
)
from bosonic_simulator.algorithms.applyunitaries import applyunitaries
from bosonic_simulator.algorithms.simulateexactly import simulateexactly

app = Flask(__name__)
CORS(app)

STATIC = "static"
os.makedirs(STATIC, exist_ok=True)


# -------------------------
# State helpers
# -------------------------
def vacuum(n):
    return GaussianStateDescription.vacuum_state(n)


def cplx(z):
    return np.complex128(z[0] + 1j * z[1])


def decode_state(s):
    return json.loads(base64.urlsafe_b64decode(s).decode())


def encode_state(s):
    return base64.urlsafe_b64encode(json.dumps(s).encode()).decode()


# -------------------------
# Gate conversion
# -------------------------
def unitary_from_gate(g):
    if g["type"] == "D":
        alpha = np.array([cplx(a) for a in g["alpha"]])
        return DisplacementDescription(alpha)
    if g["type"] == "F":
        return PhaseShiftDescription(np.float64(g["phi"]), g["mode"])
    if g["type"] == "S":
        return SqueezingDescription(np.float64(g["z"]), g["mode"])
    if g["type"] == "B":
        return BeamSplitterDescription(np.float64(g["omega"]), g["j"], g["k"])
    raise ValueError("Unknown gate")


# -------------------------
# Qiskit rendering
# -------------------------
def build_qiskit(num_wires, gates):
    qc = QuantumCircuit(num_wires)
    for g in gates:
        if g["type"] == "D":
            qc.append(Gate("D", num_wires, []), list(range(num_wires)))
        elif g["type"] == "F":
            qc.append(Gate("F", 1, [g["phi"]]), [g["mode"]])
        elif g["type"] == "S":
            qc.append(Gate("S", 1, [g["z"]]), [g["mode"]])
        elif g["type"] == "B":
            qc.append(Gate("B", 2, [g["omega"]]), [g["j"], g["k"]])
    return qc


# -------------------------
# API
# -------------------------
@app.route("/render", methods=["POST"])
def render():
    data = decode_state(request.json["state"])
    target = request.json["target"]

    if target == "global":
        gates = data["global_gates"]
        name = "global"
    else:
        gates = data["terms"][target]["gates"]
        name = f"term_{target}"

    qc = build_qiskit(data["num_wires"], gates)
    path = f"{STATIC}/{name}.png"
    circuit_drawer(qc, output="mpl", filename=path)
    return send_file(path, mimetype="image/png")


@app.route("/simulate", methods=["POST"])
def simulate():
    data = decode_state(request.json["state"])

    amplitude = np.array([cplx(a) for a in request.json["amplitude"]])
    wires = request.json["wires"]

    superposition_terms = []

    for t in data["terms"]:
        gsd = vacuum(data["num_wires"])
        unitaries = [unitary_from_gate(g) for g in t["gates"]]
        gsd = applyunitaries(gsd, unitaries)
        superposition_terms.append((cplx(t["coeff"]), gsd))

    global_unitaries = [unitary_from_gate(g) for g in data["global_gates"]]

    prob = simulateexactly(superposition_terms, global_unitaries, amplitude, wires)

    return jsonify({"probability": float(prob)})


# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
