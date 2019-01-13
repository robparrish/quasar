import quasar

def test_gates():

    I = quasar.Gate.I
    print I
    print I.U

def test_ghz_5():

    circuit = quasar.Circuit(N=5)
    circuit.add_gate(T=0, key=0, gate=quasar.Gate.H)   
    circuit.add_gate(T=1, key=(0,1), gate=quasar.Gate.CNOT)
    circuit.add_gate(T=2, key=(1,2), gate=quasar.Gate.CNOT)
    circuit.add_gate(T=3, key=(2,3), gate=quasar.Gate.CNOT)
    circuit.add_gate(T=4, key=3, gate=quasar.Gate.H)   
    circuit.add_gate(T=4, key=4, gate=quasar.Gate.H)   
    circuit.add_gate(T=5, key=(4,3), gate=quasar.Gate.CNOT)
    circuit.add_gate(T=6, key=3, gate=quasar.Gate.H)   
    circuit.add_gate(T=6, key=4, gate=quasar.Gate.H)   

    print circuit

    print circuit.compressed()
    
    print circuit.Ts
        

if __name__ == '__main__':

    test_gates()
    test_ghz_5()
