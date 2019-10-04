import torch
from torch.autograd import Function

from qiskit import QuantumRegister,QuantumCircuit,ClassicalRegister,execute
from qiskit.circuit import Parameter
from qiskit import Aer
import numpy as np


def to_numbers(tensor_list):
    num_list = []
    for tensor in tensor_list:
        num_list += [tensor.item()]
    return num_list

class QiskitCircuit():
    
    def __init__(self,create_circuit_fn, parameters, shots):
        self.parameters = parameters
        self.shots = shots       
        self.circuit = create_circuit_fn()
        
    def N_qubit_expectation_Z(self,counts, shots, nr_qubits):
        expects = np.zeros(nr_qubits)
        for key in counts.keys():
            perc = counts[key]/shots
            check = np.array([(float(key[i])-1/2)*2*perc for i in range(nr_qubits)])
            expects += check   
        return expects    
    
    def bind(self, parameters):
        self.parameters = to_numbers(parameters)
        self.circuit.data[2][0]._params = to_numbers(parameters)
    
    def run(self, i):
        self.bind(i)
        
        backend = Aer.get_backend('qasm_simulator')
        job_sim = execute(self.circuit,backend,shots=self.shots)
        result_sim = job_sim.result()
        counts = result_sim.get_counts(self.circuit)
        return self.N_qubit_expectation_Z(counts,self.shots,1)

def get_TorchCircuit(create_circuit_fn, parameters, shots=1000):   
    
    class TorchCircuit(Function):    

        @staticmethod
        def forward(ctx, i):
            if not hasattr(ctx, 'QiskitCirc'):
                ctx.QiskitCirc = QiskitCircuit(create_circuit_fn, parameters, shots)

            exp_value = ctx.QiskitCirc.run(i[0])

            result = torch.tensor([exp_value])

            ctx.save_for_backward(result, i)

            return result

        @staticmethod
        def backward(ctx, grad_output):
            eps = 0.01

            forward_tensor, i = ctx.saved_tensors    
            input_numbers = to_numbers(i[0])
            gradient = []

            for k in range(len(input_numbers)):
                input_eps = input_numbers
                input_eps[k] = input_numbers[k] + eps

                exp_value = ctx.QiskitCirc.run(torch.tensor(input_eps))[0]
                result_eps = torch.tensor([exp_value])
                gradient_result = (exp_value - forward_tensor[0][0].item())/eps
                gradient.append(gradient_result)

    #         print(gradient)
            result = torch.tensor([gradient])
    #         print(result)

            return result.float() * grad_output.float()
    
    return TorchCircuit()