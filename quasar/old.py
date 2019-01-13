import numpy as np
import collections

class Gate(object):

    def __init__(
        self,
        N,
        Ufun,
        params,
        name,
        ascii_symbols,
        ):
        
        self.N = N
        self.Ufun = Ufun
        self.params = params
        self.name = name
        self.ascii_symbols = ascii_symbols
    
    def U(self): 
        """ The (2**N,)*2 unitary matrix underlying this Gate. 

        The action of the gate on a given state is given graphically as,

        |\Psi> -G- |\Psi'>

        and mathematically as,

        |\Psi_I'> = \sum_J U_IJ |\Psi_J>

        Returns:
            (np.ndarray of shape (2**N,)*2) - the unitary matrix underlying
                this gate, built from the current parameter state.
        """
        return self.Ufun(params)

    # => Copying <= #
    
    def copy(self):
        """ Make a deep copy of the current Gate. 
        
        Returns:
            (Gate) - a copy of this Gate whose parameters may be modified
                without modifying the parameters of self.
        """
        return Gate(
            N=self.N, 
            Ufun=self.Ufun, 
            params=self.params.copy(), 
            name=self.name,  
            self.ascii_symbols=ascii_symbols,
            )

    # => Parameter Access <= #

    def set_param(self, key, param):
        """ Set the value of a parameter of this Gate. 

        Params:
            key (str) - the key of the parameter
            param (float) - the value of the parameter
        Result:
            self.params[key] = param. If the Gate does not have a parameter
                corresponding to key, a RuntimeError is thrown.
        """
        if key not in self.params: raise RuntimeError('Key %s is not in params' % key)
        self.params[key] = param

    def set_params(self, params):
        """ Set the values of multiple parameters of this Gate.

        Params:
            params (dict of str : float) -  dict of param values
        Result:
            self.params is updated with the contents of params by calling
                self.set_param for each key/value pair.
        """
        for key, param in params.iteritems():
            self.set_param(key=key, param=param)

## ==> Gates/Circuits <== ##

class Gate(object): 

    """ Class Gate represents a general N-body quantum gate. 

    An N-Body quantum gate applies a unitary operator to the state of a subset
    of N qubits, with an implicit identity matrix acting on the remaining
    qubits. The Gate class specifies the (2**N,)*2 unitary matrix for the N
    active qubits, but does not specify which qubits are active.

    There are two paths to constructing a Gate. The first is explicit, e.g.,
        I = Gate(
            N=1, 
            Uexplicit=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128), 
            name='I',
            )
    This constructs an explicit, immutable Gate with no adjustable parameters.
    The second is implicit/parametrized, e.g.,
        Rx = Gate(
            N=1,
            Ufun=Rx_fun,
            params={ 'theta' : 0.0 },
            name_fun=Rx_name_fun,
            )
    Here Ufun is a function which takes a dict of params and returns the
    unitary matrix U for the current parameter set, while name_fun is a function
    which takes a dict of params and returns the name for the current parameter
    set.
    """

    def __init__(
        self,
        N,
        Uexplicit=None,
        name=None,
        Ufun=None,
        params=None,
        name_fun=None,
        ):

        """ Initializer. Params are set as object attributes.

        Params:
            N (int) - the dimensionality of the quantum gate, e.g, 1 for
                1-body, 2 for 2-body, etc.
            Uexplicit (np.ndarray of shape (2**N,)*2 or None) - the unitary
                matrix for this gate if explicitly specified.
            name (str) - the name of this Gate if explicity specified.
            Ufun (function of dict of str : float -> np.ndarray of shape
                (2**N,)*2 or None) - a function which generates the unitary
                matrix for this Gate from the current parameter set, if an
                implicit parametrized Gate is specified.
            params (dict of str : float or None) - the dictionary of initial
                Gate parameters, if an implicit parametrized Gate is specified.
            name_fun (function of dict of str : float -> str or None) - a
                function which returns the name of this Gate from the current
                parameter set, if an implicit parametrized Gate is specified.

        A representation invariant of this class is that exlusively either
        Uexplicit/name or Ufun/params/name_fun are specified. An error is
        thrown in the initializer if this is not respected.
        """
        self.N = N
        # Choice 1: Explict construction
        self.Uexplicit = Uexplicit
        self.name = name
        # Choice 2: Mutable/parametrized construction
        self.Ufun = Ufun
        self.params = params
        self.name_fun = name_fun

        # Validity checks
        if self.is_explicit:
            # Must be set
            if self.Uexplicit is None: raise RuntimeError('Uexplicit must not be None')
            if self.name is None: raise RuntimeError('name must not be None')
            # Must not be set
            if self.Ufun is not None: raise RuntimeError('Ufun must be None')
            if self.params is not None: raise RuntimeError('params must be None')
            if self.name_fun is not None: raise RuntimeError('name_fun must be None')
        else:
            # Must not be set
            if self.Uexplicit is not None: raise RuntimeError('Uexplicit must be None')
            if self.name is not None: raise RuntimeError('name must be None')
            # Must be set
            if self.Ufun is None: raise RuntimeError('Ufun must not be None')
            if self.params is None: raise RuntimeError('params must not be None')
            if self.name_fun is None: raise RuntimeError('name_fun must not be None')
        if self.U.shape != (2**self.N,)*2: raise RuntimeError('U must be shape (2**N,)*2')

    @property
    def U(self):
        """ The (2**N,)*2 unitary matrix underlying this Gate. 

        The action of the gate on a given state is given graphically as,

        |\Psi> -G- |\Psi'>

        and mathematically as,

        |\Psi_I'> = \sum_J U_IJ |\Psi_J>

        Returns:
            (np.ndarray of shape (2**N,)*2) - the unitary matrix underlying
                this gate, build from the current parameter state.
        """
        if self.is_explicit: return self.Uexplicit
        return self.Ufun(self.params)

    def __str__(self):
        """ String representation of this Gate for the current parameter state. """
        if self.is_explicit: return self.name
        return self.name_fun(self.params)

    # => Mutability/Immutability/Copying <= #
    
    @property
    def is_explicit(self):
        """ Is this gate explicitly known (and thus immutable) or parametrized
            (and thus mutable)?
        
        Formally, this returns True if self.Uexplicit is not None else False.
        Representation invariants are checked in the constructor to ensure
        validity of either the explicit or parametrized/mutable constructor
        choices.
        """
        return True if self.Uexplicit is not None else False

    def copy(self):
        """ Make a deep copy of the current Gate, if the Gate is mutable. 
        
        Returns:
            (Gate) - a copy of this Gate whose parameters may be modified
                without modifying the parameters of self. If self is immutable
                (explicit), self is directly returned.
        """
        if self.is_explicit: return self
        return Gate(
            N=self.N, 
            Ufun=self.Ufun, 
            params=self.params.copy(), 
            name_fun=self.name_fun,  
            )

    # => Parameter Access <= #

    @property
    def param_keys(self):
        """ Return a (sorted) list of the keys of the parameters of this Gate. """
        return [] if self.is_explicit else list(sorted(self.params.keys()))

    def set_param(self, key, param):
        """ Set the value of a parameter of this Gate. 

        Params:
            key (str) - the key of the parameter
            param (float) - the value of the parameter
        Result:
            self.params[key] = param. If the Gate does not have a parameter
                corresponding to key, a RuntimeError is thrown.
        """
        if self.is_explicit: raise RuntimeError('No params in this Gate')
        if key not in self.params: raise RuntimeError('Key %s is not in params' % key)
        self.params[key] = param

    def set_params(self, params):
        """ Set the values of multiple parameters of this Gate.

        Params:
            params (dict of str : float) -  dict of param values
        Result:
            self.params is updated with the contents of params by calling
                self.set_param for each key/value pair.
        """
        for key, param in params.iteritems():
            self.set_param(key=key, param=param)

# => Gate Specialization <= #

""" I (identity) gate """
Gate.I = Gate(
    N=1,
    Ufun = lambda x : np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    name='I',
    params=collections.OrderedDict(),
    ascii_symbols=['I'],
    )

# Common 1-qubit gates (parameter free)
Gate.I = Gate(N=1, Uexplicit=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128), name='I')
Gate.X = Gate(N=1, Uexplicit=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128), name='X')
Gate.Y = Gate(N=1, Uexplicit=np.array([[0.0, -1.0j], [+1.0j, 0.0]], dtype=np.complex128), name='Y')
Gate.Z = Gate(N=1, Uexplicit=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128), name='Z')
Gate.S = Gate(N=1, Uexplicit=np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128), name='S')
Gate.T = Gate(N=1, Uexplicit=np.array([[1.0, 0.0], [0.0, np.exp(np.pi/4.0*1.j)]], dtype=np.complex128), name='T')
Gate.H = Gate(N=1, Uexplicit=1.0 / np.sqrt(2.0) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128), name='H')

# exp(-i (pi/2) * X) : Z -> Y basis transformation
Gate.Rx2 = Gate(N=1, Uexplicit=1.0 / np.sqrt(2.0) * np.array([[1.0, +1.0j], [+1.0j, 1.0]], dtype=np.complex128), name='Rx2')
Gate.Rx2T = Gate(N=1, Uexplicit=1.0 / np.sqrt(2.0) * np.array([[1.0, -1.0j], [-1.0j, 1.0]], dtype=np.complex128), name='Rx2T')

# Common 2-body gates (parameter free)
Gate.SWAP = Gate(N=2, Uexplicit=np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.complex128), name='SWAP')
Gate.CNOT = Gate(N=2, Uexplicit=np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.complex128), name='CNOT')
Gate.CZ = Gate(N=2, Uexplicit=np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0,-1.0],
    ], dtype=np.complex128), name='CZ')

# 1-body rotation gates (theta parameter)

@staticmethod
def _GateRx(theta):

    """ Rx (theta) = exp(-i * theta * x) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1.j*s], [+1.j*s, c]], dtype=np.complex128)
    
    def name_fun(params):
        return 'Rx(%.3f)' % params['theta']

    return Gate(
        N=1,
        Ufun=Ufun,
        params={'theta' : theta},
        name_fun=name_fun)
    
@staticmethod
def _GateRy(theta):

    """ Ry (theta) = exp(-i * theta * Y) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [+s, c]], dtype=np.complex128)
    
    def name_fun(params):
        return 'Ry(%.3f)' % params['theta']

    return Gate(
        N=1,
        Ufun=Ufun,
        params={'theta' : theta},
        name_fun=name_fun)

@staticmethod
def _GateRz(theta):

    """ Rz (theta) = exp(-i * theta * Z) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c-1.j*s, 0.0], [0.0, c+1.j*s]], dtype=np.complex128)
    
    def name_fun(params):
        return 'Rz(%.3f)' % params['theta']

    return Gate(
        N=1,
        Ufun=Ufun,
        params={'theta' : theta},
        name_fun=name_fun)

Gate.Rx = _GateRx
Gate.Ry = _GateRy
Gate.Rz = _GateRz

@staticmethod
def _GateSO4(A, B, C, D, E, F):
    
    def Ufun(params):
        A = params['A']
        B = params['B']
        C = params['C']
        D = params['D']
        E = params['E']
        F = params['F']
        X = np.array([
            [0.0, +A,  +B,  +C],
            [-A, 0.0,  +D,  +E],
            [-B,  -D, 0.0,  +F],
            [-C,  -E,  -F, 0.0],
            ])
        import scipy.linalg
        U = scipy.linalg.expm(X)
        return np.array(U, dtype=np.complex128)

    def name_fun(params):
        return 'SO4'
    
    return Gate(
        N=2,
        Ufun=Ufun,
        params={'A' : A, 'B' : B, 'C' : C, 'D' : D, 'E' : E, 'F' : F},
        name_fun=name_fun)

Gate.SO4 = _GateSO4

@staticmethod
def _GateSO42(thetaIY, thetaYI, thetaXY, thetaYX, thetaZY, thetaYZ):
    
    def Ufun(params):
        A = -(params['thetaIY'] + params['thetaZY'])
        F = -(params['thetaIY'] - params['thetaZY'])
        C = -(params['thetaYX'] + params['thetaXY'])
        D = -(params['thetaYX'] - params['thetaXY'])
        B = -(params['thetaYI'] + params['thetaYZ'])
        E = -(params['thetaYI'] - params['thetaYZ'])
        X = np.array([
            [0.0, +A,  +B,  +C],
            [-A, 0.0,  +D,  +E],
            [-B,  -D, 0.0,  +F],
            [-C,  -E,  -F, 0.0],
            ])
        import scipy.linalg
        U = scipy.linalg.expm(X)
        return np.array(U, dtype=np.complex128)

    def name_fun(params):
        return 'SO42'
    
    return Gate(
        N=2,
        Ufun=Ufun,
        params={
            'thetaIY' : thetaIY,
            'thetaYI' : thetaYI,
            'thetaXY' : thetaXY,
            'thetaYX' : thetaYX,
            'thetaZY' : thetaZY,
            'thetaYZ' : thetaYZ,
        },
        name_fun=name_fun)

Gate.SO42 = _GateSO42

@staticmethod
def _CF(theta):

    """ Controlled F gate """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0,  +c,  +s],
            [0.0, 0.0,  +s,  -c],
            ], dtype=np.complex128)
    
    def name_fun(params):
        return 'CF'

    return Gate(
        N=2,
        Ufun=Ufun,
        params={'theta' : theta},
        name_fun=name_fun)

Gate.CF = _CF


# Special explicit gates 

@staticmethod
def _GateU1(U):

    """ An explicit 1-body gate that is specified by the user. """

    return Gate(
        N=1,
        Uexplicit=U,
        name='U1',
        )

@staticmethod
def _GateU2(U):

    """ An explicit 2-body gate that is specified by the user. """

    return Gate(
        N=2,
        Uexplicit=U,
        name='U2',
        )

Gate.U1 = _GateU1
Gate.U2 = _GateU2

class Circuit(object):

    """ Class Circuit represents a general quantum circuit acting on N
        linearly-arranged cubits. Non-local connectivity is permitted - the
        linear arrangement is strictly for simplicity.

        An example Circuit construction is,

        circuit = Circuit(N=2)
        circuit.add_gate(T=0, key=0, Gate.H)
        circuit.add_gate(T=0, key=(1,), Gate.X)
        circuit.add_gate(T=1, key=(0,1), Gate.CNOT)
        print circuit
        
        A Circuit is always constructed with a fixed number of qubits N, but
        the time window of the circuit is freely expandable from T=0 onward.
        The Circuit starts empty, and is filled one gate at a time by the
        add_gate function.
    
        The Circuit attribute Ts (set of int) contains the set of time indices
        T with significant gates, and the Circuit attribute M (int) contains
        the total number of time moments, including empty moments.

        The core data of a Circuit is the gates attribute, which contains a
        dict of (T, key) : Gate pairs for significant gates. The (T, key)
        compound key specifies the time moment of the gate T (int), and the qubit
        connections in key (tuple of int). len(key) is always gate.N.
        """

    def __init__(
        self,
        N,
        ):

        """ Initializer.

        Params:
            N (int) - number of qubits in this circuit
        """

        self.N = N
        # All circuits must have at least one qubit
        if self.N <= 0: raise RuntimeError('N <= 0')
    
        # Primary circuit data structure
        self.gates = {} # (T, (A, [B], [C], ...)) -> Gate
        # Memoization
        self.Ts = set() # ({T}) tells unique time moments
        self.TAs = set() # ({T,A}) tells occupied circuit indices

    @property
    def M(self):
        """ The total number of time moments in the circuit (including blank moments) """
        return max(self.Ts) + 1 if len(self.Ts) else 0

    @property
    def ngate(self):
        """ The total number of gates in the circuit. """
        return len(self.gates)

    @property
    def ngate1(self):
        """ The total number of 1-body gates in the circuit. """
        return len([gate for gate in self.gates.values() if gate.N == 1])

    @property
    def ngate2(self):
        """ The total number of 2-body gates in the circuit. """
        return len([gate for gate in self.gates.values() if gate.N == 2])

    def add_gate(
        self,
        T,
        key,
        gate,
        ):

        """ Add a gate to the circuit.

        Params:
            T (int) - the time index to add the gate at
            key (int or tuple of int) - the qubit index or indices to add the gate at
            gate (Gate) - the gate to add 
        Result:
            self is updated with the added gate. Checks are performed to ensure
                that the addition is valid.

        For one body gate, can add as either of:
            circuit.add_gate(T, A, gate)
            circuit.add_gate(T, (A,), gate)
        For two body gate, must add as:
            circuit.add_gate(T, (A, B), gate)
        """

        # Make key a tuple regardless of input
        key2 = (key,) if isinstance(key, int) else key
        # Check that T >= 0
        if T < 0: raise RuntimeError('Negative T: %d' % T)
        # Check that key makes sense for gate.N
        if len(key2) != gate.N: raise RuntimeError('%d key entries provided for %d-body gate' % (len(key2), gate.N))
        # Check that the requested circuit locations are open
        for A in key2:
            if (T,A) in self.TAs: 
                raise RuntimeError('T=%d, A=%d circuit location is already occupied' % (T,A))
            if A >= self.N:
                raise RuntimeError('No qubit location %d' % A)
        # Add gate to circuit
        self.gates[(T, key2)] = gate
        # Update memoization of TAs and Ts
        for A in key2:
            self.TAs.add((T,A))
        self.Ts.add(T)

    # => Copy/Subsets/Concatenation <= #

    def copy(
        self,
        ):

        circuit = Circuit(N=self.N)
        for key, gate in self.gates.iteritems():
            T, key2 = key
            circuit.add_gate(T=T, key=key2, gate=gate.copy())
        return circuit

    def subset(
        self,
        Ts,
        copy=False,
        ):

        circuit = Circuit(N=self.N)
        for T2, Tref in enumerate(Ts):
            if Tref >= self.M: raise RuntimeError('T >= self.M: %d' % Tref)
            for key, gate in self.gates.iteritems():
                T, key2 = key
                if T == Tref:
                    circuit.add_gate(T=T2, key=key2, gate=gate.copy() if copy else gate)
        return circuit

    @staticmethod
    def concatenate(
        circuits,
        copy=False,
        ):

        if any(x.N != circuits[0].N for x in circuits): 
            raise RuntimeError('Circuits must all have same N to be concatenated')
        
        circuit = Circuit(N=circuits[0].N)
        Tstart = 0
        for circuit2 in circuits:   
            for key, gate in circuit2.gates.iteritems():
                T, key2 = key
                circuit.add_gate(T=T+Tstart, key=key2, gate=gate.copy() if copy else gate)
            Tstart += circuit2.M
        return circuit

    def deadjoin(
        self,
        As,
        copy=False,
        ):

        for A2, Aref in enumerate(As):
            if Aref >= self.N: raise RuntimeError('A >= self.A: %d' % Aref)

        Amap = { v : k for k, v in enumerate(As) }

        circuit = Circuit(N=len(As))
        for key, gate in self.gates.iteritems():
            T, key2 = key
            if all(x in Amap for x in key2):
                circuit.add_gate(T=T, key=tuple(Amap[x] for x in key2), gate=gate.copy() if copy else gate)
        return circuit

    @staticmethod
    def adjoin(
        circuits,
        copy=False,
        ):

        circuit = Circuit(N=sum(x.N for x in circuits))
        Astart = 0
        for circuit2 in circuits:   
            for key, gate in circuit2.gates.iteritems():
                T, key2 = key
                circuit.add_gate(T=T, key=tuple(x + Astart for x in key2), gate=gate.copy() if copy else gate)
            Astart += circuit2.N
        return circuit
    
    def reversed(
        self,
        copy=False,
        ):

        circuit = Circuit(N=self.N)
        for key, gate in self.gates.iteritems():
            T, key2 = key
            circuit.add_gate(T=self.M-T-1, key=key2, gate=gate)
        return circuit

    def nonredundant(
        self,
        copy=False,
        ):

        circuit = Circuit(N=self.N)
        Tmap = { v : k for k, v in enumerate(sorted(self.Ts)) }
        for key, gate in self.gates.iteritems():
            T, key2 = key
            circuit.add_gate(T=Tmap[T], key=key2, gate=gate)
        return circuit

    def compressed(
        self,
        ):

        """ Return an equivalent circuit with 1/2-body gates merged together to
            minimize the number of gates by using composite 1- and 2-body gate
            operations.
    
        """

        # Jam consecutive 1-body gates (removes runs of 1-body gates)
        circuit1 = self.copy()
        plan = [[0 for x in range(self.M)] for y in range(self.N)]
        for key, gate in circuit1.gates.iteritems():
            T, key2 = key
            if gate.N == 1:
                A, = key2
                plan[A][T] = 1
            elif gate.N == 2:
                A, B = key2
                plan[A][T] = 2
                plan[B][T] = -2
            else:
                raise RuntimeError("N > 2")
        circuit2 = Circuit(N=self.N)
        for A, row in enumerate(plan):
            Tstar = None
            U = None
            for T, V in enumerate(row):
                # Start the 1-body gate chain
                if V == 1 and U is None:
                    Tstar = T
                    U = np.copy(circuit1.gates[T,(A,)].U)
                # Continue the 1-body gate chain
                elif V == 1:
                    U = np.dot(circuit1.gates[T,(A,)].U, U)
                # If 2-body gate or end of circuit encountered, place 1-body gate
                if U is not None and (V == 2 or V == -2 or T == self.M - 1):
                    circuit2.add_gate(T=Tstar, key=(A,), gate=Gate.U1(U=U))
                    Tstar = None
                    U = None
        for key, gate in circuit1.gates.iteritems():
            T, key2 = key
            if gate.N == 2:
                circuit2.add_gate(T=T, key=key2, gate=gate)

        # Jam 1-body gates into 2-body gates if possible (not possible if 1-body gate wire)
        circuit1 = circuit2
        plan = [[0 for x in range(self.M)] for y in range(self.N)]
        for key, gate in circuit1.gates.iteritems():
            T, key2 = key
            if gate.N == 1:
                A, = key2
                plan[A][T] = 1
            elif gate.N == 2:
                A, B = key2
                plan[A][T] = 2
                plan[B][T] = -2
            else:
                raise RuntimeError("N > 2")
        circuit2 = Circuit(N=self.N)
        jammed_gates = {}                 
        for key, gate in circuit1.gates.iteritems():
            if gate.N != 2: continue
            T, key2 = key
            A, B = key2
            U = np.copy(gate.U)
            # Left-side 1-body gates
            for T2 in range(T-1,-1,-1):
                if plan[A][T2] == 2 or plan[A][T2] == -2: break
                if plan[A][T2] == 1:
                    gate1 = circuit1.gates[T2, (A,)]
                    U = np.dot(U, np.kron(gate1.U, np.eye(2)))
                    jammed_gates[T2, (A,)] = gate1
                    break
            for T2 in range(T-1,-1,-1):
                if plan[B][T2] == 2 or plan[B][T2] == -2: break
                if plan[B][T2] == 1:
                    gate1 = circuit1.gates[T2, (B,)]
                    U = np.dot(U, np.kron(np.eye(2), gate1.U))
                    jammed_gates[T2, (B,)] = gate1
                    break
            # Right-side 1-body gates (at circuit end)
            if T+1 < self.M and max(abs(plan[A][T2]) for T2 in range(T+1, self.M)) == 1:
                T2 = [T3 for T3, P in enumerate(plan[A][T+1:self.M]) if P == 1][0] + T+1
                gate1 = circuit1.gates[T2, (A,)]
                U = np.dot(np.kron(gate1.U, np.eye(2)), U)
                jammed_gates[T2, (A,)] = gate1
            if T+1 < self.M and max(abs(plan[B][T2]) for T2 in range(T+1, self.M)) == 1:
                T2 = [T3 for T3, P in enumerate(plan[B][T+1:self.M]) if P == 1][0] + T+1
                gate1 = circuit1.gates[T2, (B,)]
                U = np.dot(np.kron(np.eye(2), gate1.U), U)
                jammed_gates[T2, (B,)] = gate1
            circuit2.add_gate(T=T, key=key2, gate=Gate.U2(U=U))
        # Unjammed gates (should all be 1-body on 1-body wires) 
        for key, gate in circuit1.gates.iteritems():
            if gate.N != 1: continue
            T, key2 = key
            if key not in jammed_gates:
                circuit2.add_gate(T=T, key=key2, gate=gate)

        # Jam 2-body gates, if possible
        circuit1 = circuit2
        circuit2 = Circuit(N=self.N)
        jammed_gates = {}
        for T in range(circuit1.M):
            circuit3 = circuit1.subset([T])
            for key, gate in circuit3.gates.iteritems():
                if gate.N != 2: continue
                T4, key2 = key
                if (T, key2) in jammed_gates: continue
                A, B = key2
                jams = [((T, key2), gate, False)]
                for T2 in range(T+1, self.M):
                    if (T2, (A, B)) in circuit1.gates:
                        jams.append(((T2, (A, B)), circuit1.gates[(T2, (A, B))], False))
                    elif (T2, (B, A)) in circuit1.gates:
                        jams.append(((T2, (B, A)), circuit1.gates[(T2, (B, A))], True))
                    elif (T2, A) in circuit1.TAs:
                        break # Interference
                    elif (T2, B) in circuit1.TAs:
                        break # Interference
                U = np.copy(jams[0][1].U)
                for idx in range(1, len(jams)):
                    key, gate, trans = jams[idx]
                    U2 = np.copy(gate.U)
                    if trans:
                        U2 = np.reshape(np.einsum('ijkl->jilk', np.reshape(U2, (2,)*4)), (4,)*2)
                    U = np.dot(U2,U)
                circuit2.add_gate(T=T, key=(A,B), gate=Gate.U2(U=U))
                for key, gate, trans in jams:
                    jammed_gates[key] = gate
        # Unjammed gates (should all be 1-body on 1-body wires)
        for key, gate in circuit1.gates.iteritems():
            if gate.N != 1: continue
            T, key2 = key
            if key not in jammed_gates:
                circuit2.add_gate(T=T, key=key2, gate=gate)

        return circuit2.nonredundant()

    # => Parameter Access <= #

    @property
    def params(self):
        """ A dict of (T, key, param_name) : param_value for all mutable parameters in the circuit. """ 
        params = {}
        for key, gate in self.gates.iteritems():
            T, key2 = key
            if gate.is_explicit: continue
            for k, v in gate.params.iteritems():
                params[(T, key2, k)] = v
        return params

    def set_params(
        self,
        params,
        ):

        for k, v in params.iteritems():
            T, key2, name = k
            self.gates[(T, key2)].set_param(key=name, param=v)

    # => ASCII Circuit Diagrams <= #

    def __str__(
        self,
        ):
        
        return self.diagram(time_lines='both')

    def diagram(
        self,
        time_lines='both',
        ):

        # Left side states
        Wd = int(np.ceil(np.log10(self.N)))
        lstick = '%-*s : |\n' % (2+Wd, 'T')
        for x in range(self.N): 
            lstick += '%*s\n' % (6+Wd, ' ')
            lstick += '|%*d> : -\n' % (Wd, x)

        # Build moment strings
        moments = []
        for T in range(self.M):
            moments.append(self.diagram_moment(
                T=T,
                adjust_for_T=False if time_lines=='neither' else True,
                ))

        # Unite strings
        lines = lstick.split('\n')
        for moment in moments:
            for i, tok in enumerate(moment.split('\n')):
                lines[i] += tok
        # Time on top and bottom
        lines.append(lines[0])

        # Adjust for time lines
        if time_lines == 'both':
            pass
        elif time_lines == 'top':
            lines = lines[:-2]
        elif time_lines == 'bottom':
            lines = lines[2:]
        elif time_lines == 'neither':
            lines = lines[2:-2]
        else:
            raise RuntimeError('Invalid time_lines argument: %s' % time_lines)
        
        strval = '\n'.join(lines)

        return strval

    def diagram_moment(
        self,
        T,
        adjust_for_T=True,
        ):

        circuit = self.subset([T])

        two_body_symbols = {
            'SWAP' : ('X', 'X'),
            'CNOT' : ('@', 'X'),
            'CZ'   : ('@', 'Z'),
            'CF'   : ('@', 'F'),
            'U2'   : ('U2A', 'U2B'),
            'SO4'   : ('SO4A', 'SO4B'),
            'SO42'   : ('SO42A', 'SO42B'),
        }

        # list (total seconds) of dict of A -> gate symbol
        seconds = [{}]
        # list (total seconds) of dict of A -> interstitial symbol
        seconds2 = [{}]
        for key, gate in circuit.gates.iteritems():
            T2, key2 = key
            # Find the first second this gate fits within (or add a new one)
            for idx, second in enumerate(seconds):
                fit = not any(A in second for A in range(min(key2), max(key2)+1))
                if fit:
                    break
            if not fit:
                seconds.append({})
                seconds2.append({})
                idx += 1
            # Put the gate into that second
            for A in range(min(key2), max(key2)+1):
                # Gate symbol
                if A in key2:
                    if gate.N == 1:
                        seconds[idx][A] = str(gate)
                    elif gate.N == 2:
                        Aind = [Aind for Aind, B in enumerate(key2) if A == B][0]
                        seconds[idx][A] = two_body_symbols[str(gate)][Aind]
                    else:
                        raise RuntimeError('Unkown N>2 gate')
                else:
                    seconds[idx][A] = '|'
                # Gate connector
                if A != min(key2):
                    seconds2[idx][A] = '|'

        # + [1] for the - null character
        wseconds = [max([len(v) for k, v in second.iteritems()] + [1]) for second in seconds]
        wtot = sum(wseconds)    

        # Adjust widths for T field
        Tsymb = '%d' % T
        if adjust_for_T:
            if wtot < len(Tsymb): wseconds[0] += len(Tsymb) - wtot
            wtot = sum(wseconds)    
        
        Is = ['' for A in range(self.N)]
        Qs = ['' for A in range(self.N)]
        for second, second2, wsecond in zip(seconds, seconds2, wseconds):
            for A in range(self.N):
                Isymb = second2.get(A, ' ')
                IwR = wsecond - len(Isymb)
                Is[A] += Isymb + ' ' * IwR + ' '
                Qsymb = second.get(A, '-')
                QwR = wsecond - len(Qsymb)
                Qs[A] += Qsymb + '-' * QwR + '-'

        strval = Tsymb + ' ' * (wtot + len(wseconds) - len(Tsymb) - 1) + '|\n' 
        for I, Q in zip(Is, Qs):
            strval += I + '\n'
            strval += Q + '\n'

        return strval

    def latex_diagram(
        self,
        row_params='@R=1.0em',
        col_params='@C=1.0em',
        size_params='',
        use_lstick=True,
        one_body_printing='pretty',
        variable_printing=True,
        ):

        strval = ''

        # Header
        strval += '\\begin{equation}\n'
        strval += '\\Qcircuit %s %s %s {\n' % (
            row_params,
            col_params,
            size_params,
            )

        # Qubit lines
        lines = ['' for _ in range(self.N)]

        # Lstick  
        if use_lstick:
            for A in range(self.N):
                lines[A] += '\\lstick{|%d\\rangle}\n' % A

        # Moment contents
        for T in range(self.M):
            lines2 = self.latex_diagram_moment(
                T=T,    
                one_body_printing=one_body_printing,
                variable_printing=variable_printing,
                )
            for A in range(self.N):
                lines[A] += lines2[A]
        
        # Trailing wires
        for A in range(self.N):
            lines[A] += ' & \\qw \\\\\n'

        # Concatenation
        strval += ''.join(lines)

        # Footer
        strval += '}\n'
        strval += '\\end{equation}\n'

        return strval

    def latex_diagram_moment(
        self,
        T,
        one_body_printing='pretty',
        variable_printing=True,
        ):

        circuit = self.subset([T])

        # list (total seconds) of dict of A -> gate symbol
        seconds = [{}]
        for key, gate in circuit.gates.iteritems():
            T2, key2 = key
            # Find the first second this gate fits within (or add a new one)
            for idx, second in enumerate(seconds):
                fit = not any(A in second for A in range(min(key2), max(key2)+1))
                if fit:
                    break
            if not fit:
                seconds.append({})
                idx += 1
            # Place gate lines in circuit
            if gate.N == 1:
                A, = key2
                # One-body rotation gates can be easily cleaned up
                if one_body_printing == 'plain':
                    Qstr = str(gate)
                elif one_body_printing == 'pretty':
                    Qstr = str(gate)
                    # Try for rotation gate
                    mobj = re.match(r'^R([xyz])\((\S+)\)$', Qstr)
                    if mobj:
                        if variable_printing:
                            Qstr = 'R_%s (%s)' % (mobj.group(1), mobj.group(2))
                        else:
                            Qstr = 'R_%s' % mobj.group(1)
                seconds[idx][A] = ' & \\gate{%s}\n' % Qstr
            elif gate.N == 2:
                A, B = key2
                if str(gate) == 'CNOT':
                    seconds[idx][A] = ' & \\ctrl{%d}\n' % (B-A) 
                    seconds[idx][B] = ' & \\targ\n'
                if str(gate) == 'CZ':
                    seconds[idx][A] = ' & \\ctrl{%d}\n' % (B-A) 
                    seconds[idx][B] = ' & \\gate{Z}\n'
                elif str(gate) == 'SWAP':
                    seconds[idx][A] = ' & \\qswap \\qwx[%d]\n' % (B-A) 
                    seconds[idx][B] = ' & \\qswap\n'
                elif str(gate) == 'U2':
                    seconds[idx][A] = ' & \\gate{U_2^A} \\qwx[%d]\n' % (B-A) 
                    seconds[idx][B] = ' & \\gate{U_2^B}\n'
                else:
                    raise RuntimeError('Unknown 2-body gate: %s' % gate)
            else:
                raise RuntimeError('Unknown N>2 body gate: %s' % gate)

        Qs = ['' for A in range(self.N)]
        for second in seconds:
            for A in range(self.N):
                Qs[A] += second.get(A, ' & \\qw \n')

        return Qs

    # => Simulation! <= #

    def simulate(
        self,
        wfn=None,
        dtype=np.complex128,
        ):

        """ Propagate wavefunction wfn through this circuit. 

        Params:
            wfn (np.ndarray of shape (2**self.N,) and a complex dtype or None)
                - the initial wavefunction. If None, the reference state
                \prod_{A} |0_A> will be used.
            dtype (complex dtype) - the dtype to allocate the reference state
                at if wfn is None.
        Returns:
            (np.ndarray of shape (2**self.N,) and a complex dtype) - the
                propagated wavefunction. Note that the input wfn is not
                changed by this operation.
        """

        for T, wfn in self.simulate_steps(wfn, dtype=dtype):
            pass

        return wfn

    def simulate_steps(
        self,
        wfn=None,
        dtype=np.complex128,
        ):

        """ Generator to propagate wavefunction wfn through the circuit one
            moment at a time.

        This is often used as:
        
        for T, wfn1 in simulate_steps(wfn=wfn0):
            print wfn1

        Note that to prevent repeated allocations of (2**N) arrays, this
        operation allocates two (2**N) working copies, and swaps between them
        as gates are applied. References to one of these arrays are returned at
        each moment. Therefore, if you want to save a history of the
        wavefunction, you will need to copy the wavefunction returned at each
        moment by this generator. Note that the input wfn is not changed by
        this operation.

        Params:
            wfn (np.ndarray of shape (2**self.N,) and a complex dtype or None)
                - the initial wavefunction. If None, the reference state
                \prod_{A} |0_A> will be used.
            dtype (complex dtype) - the dtype to allocate the reference state
                at if wfn is None.
        Returns (at each yield):
            (int, np.ndarray of shape (2**self.N,) and a complex dtype) - the
                time moment and current state of the wavefunction at each step
                along the propagation. Note that the input wfn is not
                changed by this operation.
        """

        # Reference state \prod_A |0_A>
        if wfn is None:
            wfn = np.zeros((2**self.N,), dtype=dtype)
            wfn[0] = 1.0

        # Don't modify user data, but don't copy all the time
        wfn1 = np.copy(wfn)
        wfn2 = np.zeros_like(wfn1)

        for T in range(self.M):
            circuit = self.subset([T])
            for key, gate in circuit.gates.iteritems():
                T, key2 = key
                if gate.N == 1:
                    wfn2 = Circuit.apply_gate_1(
                        wfn1=wfn1,
                        wfn2=wfn2,
                        U=gate.U,
                        A=key2[0],
                        )
                elif gate.N == 2:
                    wfn2 = Circuit.apply_gate_2(
                        wfn1=wfn1,
                        wfn2=wfn2,
                        U=gate.U,
                        A=key2[0],
                        B=key2[1],
                        )
                else:
                    raise RuntimeError('Cannot apply gates with N > 2: %s' % gate)
                wfn1, wfn2 = wfn2, wfn1
            yield T, wfn1

    @staticmethod
    def apply_gate_1(
        wfn1,
        wfn2,
        U,
        A,
        ):

        """ Apply a 1-body gate unitary U to wfn1 at qubit A, yielding wfn2.

        The formal operation performed is,

            wfn1_LiR = \sum_{j} U_ij wfn2_LjR

        Here L are the indices of all of the qubits to the left of A (<A), and
        R are the indices of all of the qubits to the right of A (>A).

        This function requires the user to supply both the initial state in
        wfn1 and an array wfn2 to place the result into. This allows this
        function to apply the gate without any new allocations or scratch arrays.

        Params:
            wfn1 (np.ndarray of shape (2**self.N,) and a complex dtype)
                - the initial wavefunction. Unaffected by the operation
            wfn2 (np.ndarray of shape (2**self.N,) and a complex dtype)
                - an array to write the new wavefunction into. Overwritten by
                the operation.
            U (np.ndarray of shape (2,2) and a complex dtype) - the matrix
                representation of the 1-body gate.
            A (int) - the qubit index to apply the gate at.
        Result:
            the data of wfn2 is overwritten with the result of the operation.
        Returns:
            reference to wfn2, for chaining
        """

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if U.shape != (2,2): raise RuntimeError('1-body gate must be (2,2)')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))

        L = 2**(A)     # Left hangover
        R = 2**(N-A-1) # Right hangover
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,R)
        wfn2v.shape = (L,2,R)
        np.einsum('LjR,ij->LiR', wfn1v, U, out=wfn2v)

        return wfn2

    @staticmethod
    def apply_gate_2(
        wfn1,
        wfn2,
        U,
        A,
        B,
        ):

        """ Apply a 2-body gate unitary U to wfn1 at qubits A and B, yielding wfn2.

        The formal operation performed is (for the case that A < B),

            wfn1_LiMjR = \sum_{lk} U_ijkl wfn2_LiMjR

        Here L are the indices of all of the qubits to the left of A (<A), M M
        are the indices of all of the qubits to the right of A (>A) and left of
        B (<B), and R are the indices of all of the qubits to the right of B
        (>B). If A > B, permutations of A and B and the gate matrix U are
        performed to ensure that the gate is applied correctly.

        This function requires the user to supply both the initial state in
        wfn1 and an array wfn2 to place the result into. This allows this
        function to apply the gate without any new allocations or scratch arrays.

        Params:
            wfn1 (np.ndarray of shape (2**self.N,) and a complex dtype)
                - the initial wavefunction. Unaffected by the operation
            wfn2 (np.ndarray of shape (2**self.N,) and a complex dtype)
                - an array to write the new wavefunction into. Overwritten by
                the operation.
            U (np.ndarray of shape (4,4) and a complex dtype) - the matrix
                representation of the 1-body gate. This should be packed to
                operate on the product state |A> otimes |B>, as usual.
            A (int) - the first qubit index to apply the gate at.
            B (int) - the second qubit index to apply the gate at.
        Result:
            the data of wfn2 is overwritten with the result of the operation.
        Returns:
            reference to wfn2, for chaining
        """

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if B >= N: raise RuntimeError('B >= N')
        if A == B: raise RuntimeError('A == B')
        if U.shape != (4,4): raise RuntimeError('2-body gate must be (4,4)')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))

        U2 = np.reshape(U, (2,2,2,2))
        if A > B:
            A2, B2 = B, A
            U2 = np.einsum('ijkl->jilk', U2)
        else:
            A2, B2 = A, B

        L = 2**(A2)      # Left hangover
        M = 2**(B2-A2-1) # Middle hangover
        R = 2**(N-B2-1)  # Right hangover
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,M,2,R)
        wfn2v.shape = (L,2,M,2,R)
        np.einsum('LkMlR,ijkl->LiMjR', wfn1v, U2, out=wfn2v)

        return wfn2

    @staticmethod
    def compute_1pdm(
        wfn1,
        wfn2,
        A,
        ):

        """ Compute the 1pdm (one-particle density matrix) at qubit A. 

        The 1pdm is formally defined as,

            D_ij = \sum_{L,R} wfn1_LiR^* wfn2_LjR
        
        Here L are the indices of all of the qubits to the left of A (<A), and
        R are the indices of all of the qubits to the right of A (>A).

        If wfn1 is equivalent to wfn2, a Hermitian density matrix will be
        returned. If wfn1 is not equivalent to wfn2, a non-Hermitian transition
        density matrix will be returned (the latter cannot be directly observed
        in a quantum computer, but is a very useful conceptual quantity).

        Params:
            wfn1 (np.ndarray of shape (self.N**2,) and a complex dtype) - the bra wavefunction.
            wfn2 (np.ndarray of shape (self.N**2,) and a complex dtype) - the ket wavefunction.
            A (int) - the index of the qubit to evaluate the 1pdm at
        Returns:
            (np.ndarray of shape (2,2) and complex dtype) - the 1pdm
        """

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))

        L = 2**(A)     # Left hangover
        R = 2**(N-A-1) # Right hangover
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,R)
        wfn2v.shape = (L,2,R)
        D = np.einsum('LiR,LjR->ij', wfn1v.conj(), wfn2v)
        return D

    @staticmethod
    def compute_2pdm(
        wfn1,
        wfn2,
        A,
        B,
        ):

        """ Compute the 2pdm (two-particle density matrix) at qubits A and B. 

        The formal operation performed is (for the case that A < B),

            D_ijkl = \sum_{LMR} wfn1_LiMjR^* wfn2_LkMlR

        Here L are the indices of all of the qubits to the left of A (<A), M M
        are the indices of all of the qubits to the right of A (>A) and left of
        B (<B), and R are the indices of all of the qubits to the right of B
        (>B). If A > B, permutations of A and B and the resultant 2pdm are
        performed to ensure that the 2pdm is computed correctly.

        If wfn1 is equivalent to wfn2, a Hermitian density matrix will be
        returned. If wfn1 is not equivalent to wfn2, a non-Hermitian transition
        density matrix will be returned (the latter cannot be directly observed
        in a quantum computer, but is a very useful conceptual quantity).

        Params:
            wfn1 (np.ndarray of shape (self.N**2,) and a complex dtype) - the bra wavefunction.
            wfn2 (np.ndarray of shape (self.N**2,) and a complex dtype) - the ket wavefunction.
            A (int) - the index of the first qubit to evaluate the 2pdm at
            B (int) - the index of the second qubit to evaluate the 2pdm at
        Returns:
            (np.ndarray of shape (4,4) and complex dtype) - the 2pdm in the 
                |A> otimes |B> basis.
        """

        N = (wfn1.shape[0]&-wfn1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if B >= N: raise RuntimeError('B >= N')
        if A == B: raise RuntimeError('A == B')
        if wfn1.shape != (2**N,): raise RuntimeError('wfn1 should be (%d,) shape, is %r shape' % (2**N, wfn1.shape))
        if wfn2.shape != (2**N,): raise RuntimeError('wfn2 should be (%d,) shape, is %r shape' % (2**N, wfn2.shape))

        if A > B:
            A2, B2 = B, A
        else:
            A2, B2 = A, B

        L = 2**(A2)      # Left hangover
        M = 2**(B2-A2-1) # Middle hangover
        R = 2**(N-B2-1)  # Right hangover
        wfn1v = wfn1.view() 
        wfn2v = wfn2.view()
        wfn1v.shape = (L,2,M,2,R)
        wfn2v.shape = (L,2,M,2,R)
        D = np.einsum('LiMjR,LkMlR->ijkl', wfn1v.conj(), wfn2v)
    
        if A > B:
            D = np.einsum('ijkl->jilk', D)

        return np.reshape(D, (4,4))

    @staticmethod
    def compute_pauli_1(
        wfn,
        A,
        ):

        """ Compute the expectation values of the 1-body Pauli operators at qubit A.

        E.g., the expectation value of the Z operator at qubit A is,

            <Z_A> = <wfn|\hat Z_A|wfn>

        These can be efficiently computed from the 1pdm (they are just an
            alternate representation of the 1pdm).

        Params:
            wfn (np.ndarray of shape (self.N**2,) and a complex dtype) - the wavefunction.
            A (int) - the index of the qubit to evaluate the Pauli measurements at.
        Returns:
            (np.ndarray of shape (4,) and real dtype corresponding to precision
                of wfn dtype) - the Pauli expectation values packed as [I,X,Y,Z].
        """

        D = Circuit.compute_1pdm(
            wfn1=wfn,
            wfn2=wfn,
            A=A,
            )

        I = (D[0,0] + D[1,1]).real
        Z = (D[0,0] - D[1,1]).real
        X = (D[1,0] + D[0,1]).real
        Y = (D[1,0] - D[0,1]).imag
        return np.array([I,X,Y,Z])

    @staticmethod
    def compute_pauli_2(
        wfn,
        A,
        B,
        ):

        """ Compute the expectation values of the 2-qubit Pauli operators at
            qubits A and B.

        E.g., the expectation value of the Z operator at qubit A and the X
        operator at qubit B is,

            <Z_A X_B> = <wfn|\hat Z_A \hat X_B|wfn>

        These can be efficiently computed from the 2pdm (they are just an
            alternate representation of the 2pdm).

        Params:
            wfn (np.ndarray of shape (self.N**2,) and a complex dtype) - the wavefunction.
            A (int) - the index of the first qubit to evaluate the Pauli measurements at.
            B (int) - the index of the second qubit to evaluate the Pauli measurements at.
        Returns:
            (np.ndarray of shape (4,4) and real dtype corresponding to precision
                of wfn dtype) - the Pauli expectation values packed as [I,X,Y,Z].
        """

        D = Circuit.compute_2pdm(
            wfn1=wfn,
            wfn2=wfn,
            A=A,
            B=B,
            )

        # I, X, Y, Z
        Pmats = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, -1.0j], [+1.0j, 0.0]]),
            np.array([[1.0, 0.0], [0.0, -1.0]]),
            ]

        G = np.zeros((4,4))
        for A, PA in enumerate(Pmats):
            for B, PB in enumerate(Pmats):
                G[A,B] = np.sum(np.kron(PA, PB).conj() * D).real

        return G

## ==> Test Code <== ##

def test_1():

    import time

    start = time.time()
    circuit = Circuit(N=4)
    circuit.add_gate(T=1, key=(0,1), gate=Gate.CNOT)
    circuit.add_gate(T=1, key=(2,3), gate=Gate.CNOT)
    circuit.add_gate(T=2, key=(0,3), gate=Gate.CNOT)
    circuit.add_gate(T=2, key=(2,1), gate=Gate.CNOT)
    circuit.add_gate(T=3, key=(0,2), gate=Gate.CNOT)
    circuit.add_gate(T=3, key=(3,1), gate=Gate.CNOT)
    circuit.add_gate(T=4, key=(0,3), gate=Gate.SWAP)
    circuit.add_gate(T=4, key=2, gate=Gate.Rx(theta=0.1))
    # circuit.add_gate(T=20, key=3, gate=Gate.X)
    # print '%11.3E' % (time.time() - start)

    

    # start = time.time()
    print circuit
    # print '%11.3E' % (time.time() - start)

    # start = time.time()
    # print circuit.simulate()
    # print '%11.3E' % (time.time() - start)

    print circuit.latex_diagram(
        row_params='@R=0.7em',
        col_params='@C=1.0em',
        size_params='@!R',
        variable_printing=False,
        use_lstick=True,
        )

def test_ghz_5():

    circuit = Circuit(N=5)
    circuit.add_gate(T=0, key=0, gate=Gate.H)   
    circuit.add_gate(T=1, key=(0,1), gate=Gate.CNOT)
    circuit.add_gate(T=2, key=(1,2), gate=Gate.CNOT)
    circuit.add_gate(T=3, key=(2,3), gate=Gate.CNOT)
    circuit.add_gate(T=4, key=3, gate=Gate.H)   
    circuit.add_gate(T=4, key=4, gate=Gate.H)   
    circuit.add_gate(T=5, key=(4,3), gate=Gate.CNOT)
    circuit.add_gate(T=6, key=3, gate=Gate.H)   
    circuit.add_gate(T=6, key=4, gate=Gate.H)   

    print circuit
        
    print circuit.latex_diagram(
        row_params='@R=0.7em',
        col_params='@C=1.0em',
        size_params='@!R',
        variable_printing=False,
        use_lstick=True,
        )

    circuit2 = Circuit.concatenate([circuit]*3, copy=True)
    print circuit2

    print circuit2.diagram(time_lines='top')
    print circuit2.diagram(time_lines='bottom')
    print circuit2.diagram(time_lines='neither')

def test_simulate():

    import time
    
    start = time.time()

    circuit = Circuit(N=3)
    circuit.add_gate(T=0, key=0, gate=Gate.H)
    circuit.add_gate(T=1, key=1, gate=Gate.H)
    circuit.add_gate(T=2, key=2, gate=Gate.H)
    print circuit
    wfn = circuit.simulate()

    circuit = Circuit(N=3)
    circuit.add_gate(T=0, key=0, gate=Gate.H)
    circuit.add_gate(T=0, key=1, gate=Gate.H)
    circuit.add_gate(T=0, key=2, gate=Gate.H)
    print circuit
    wfn = circuit.simulate()

    circuit = Circuit(N=2)
    circuit.add_gate(T=0, key=0, gate=Gate.H)
    circuit.add_gate(T=1, key=(0,1), gate=Gate.CNOT)
    print circuit
    wfn = circuit.simulate()
    circuit = Circuit(N=2)
    circuit.add_gate(T=0, key=0, gate=Gate.H)
    circuit.add_gate(T=1, key=0, gate=Gate.H)
    circuit.add_gate(T=1, key=1, gate=Gate.H)
    circuit.add_gate(T=2, key=(1,0), gate=Gate.CNOT)
    circuit.add_gate(T=3, key=0, gate=Gate.H)
    circuit.add_gate(T=3, key=1, gate=Gate.H)
    print circuit
    wfn = circuit.simulate()

    circuit = Circuit.concatenate([circuit,]*50)
    print circuit.diagram(time_lines='neither')
    
    steps = []
    for T, wfn2 in circuit.simulate_steps():
        steps.append(wfn2.copy())
    for step in steps:
        print step

    idx = 1
    D1 = Circuit.compute_1pdm(
        steps[idx],
        steps[idx],
        A=1,
        )
    D2 = Circuit.compute_2pdm(
        steps[idx],
        steps[idx],
        A=0,
        B=1,
        )
        
    print D1
    print D2

    V1 = Circuit.compute_pauli_1(
        steps[idx],
        A=1,
        )
    V2 = Circuit.compute_pauli_2(
        steps[idx],
        A=0,
        B=1,
        )
    
    print V1
    print V2
        
    circuit = Circuit(N=18)
    circuit.add_gate(T=0, key=0, gate=Gate.H)
    circuit.add_gate(T=1, key=0, gate=Gate.H)
    circuit.add_gate(T=1, key=1, gate=Gate.H)
    circuit.add_gate(T=2, key=(1,0), gate=Gate.CNOT)
    circuit.add_gate(T=3, key=0, gate=Gate.H)
    circuit.add_gate(T=3, key=1, gate=Gate.H)
    print circuit
    wfn = circuit.simulate()

    print '%11.3E' % (time.time() - start)


if __name__ == '__main__':

    # test_1()
    # test_ghz_5()
    test_simulate()
