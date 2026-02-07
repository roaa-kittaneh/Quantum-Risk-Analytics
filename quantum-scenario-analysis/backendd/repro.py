import numpy as np
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_aer.primitives import Sampler
from qiskit import QuantumCircuit

def test_config(mu_target, sigma_target):
    qubits = 5
    # Calculate mu, sigma for underlying normal to get LogNormal centered at mu_target
    # Simplified: mu_normal = ln(mu_target)
    mu_normal = np.log(mu_target)
    sigma_normal = sigma_target / mu_target # rough approx for small sigma
    
    # Range should cover mu_target
    low = max(0.0, mu_target - 3 * sigma_target)
    high = mu_target + 3 * sigma_target
    
    print(f"Testing target mu={mu_target}, sigma={sigma_target}")
    print(f"Normal params mu={mu_normal:.4f}, sigma={sigma_normal:.4f}")
    print(f"Bounds: [{low:.4f}, {high:.4f}]")
    
    try:
        dist = LogNormalDistribution(qubits, mu=mu_normal, sigma=sigma_normal, bounds=(low, high))
        f_obj = LinearAmplitudeFunction(qubits, slope=[1], offset=[0], domain=(low, high), image=(0, 1))
        circuit = QuantumCircuit(qubits + 1)
        circuit.append(dist, range(qubits))
        circuit.append(f_obj, range(qubits + 1))
        problem = EstimationProblem(state_preparation=circuit, objective_qubits=[qubits], post_processing=f_obj.post_processing)
        ae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, sampler=Sampler())
        result = ae.estimate(problem)
        print(f"Result: {result.estimation_processed:.6f}")
    except Exception as e:
        print(f"Error: {e}")

print("--- Testing mild ---")
test_config(0.015, 0.005)
print("\n--- Testing baseline ---")
test_config(0.010, 0.010)
