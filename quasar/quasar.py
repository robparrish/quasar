import numpy as np
import collections

# => Matrix class <= #

class Matrix(object):

    """ Class Matrix holds several common matrices encountered in quantum circuits.

    These matrices are stored in np.ndarray with dtype=np.complex128.

    The naming/ordering of the matrices in Quasar follows that of Nielsen and
    Chuang, *except* that rotation matrices are specfied in full turns:

        Rz(theta) = exp(-i*theta*Z)
    
    whereas Nielsen and Chuang define these in half turns:

        Rz^NC(theta) = exp(-i*theta*Z/2)
    """

    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Y = np.array([[0.0, -1.0j], [+1.0j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    T = np.array([[1.0, 0.0], [0.0, np.exp(np.pi/4.0*1.j)]], dtype=np.complex128)
    H = 1.0 / np.sqrt(2.0) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    # exp(-i (pi/2) * X) : Z -> Y basis transformation
    Rx2 = 1.0 / np.sqrt(2.0) * np.array([[1.0, +1.0j], [+1.0j, 1.0]], dtype=np.complex128)
    Rx2T = 1.0 / np.sqrt(2.0) * np.array([[1.0, -1.0j], [-1.0j, 1.0]], dtype=np.complex128)

    II = np.kron(I, I)
    IX = np.kron(I, X)
    IY = np.kron(I, Y)
    IZ = np.kron(I, Z)
    XI = np.kron(X, I)
    XX = np.kron(X, X)
    XY = np.kron(X, Y)
    XZ = np.kron(X, Z)
    YI = np.kron(Y, I)
    YX = np.kron(Y, X)
    YY = np.kron(Y, Y)
    YZ = np.kron(Y, Z)
    ZI = np.kron(Z, I)
    ZX = np.kron(Z, X)
    ZY = np.kron(Z, Y)
    ZZ = np.kron(Z, Z)

    CX = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.complex128)
    CY = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        [0.0, 0.0, +1.0j, 0.0],
        ], dtype=np.complex128)
    CZ = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        ], dtype=np.complex128)
    CS = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0j],
        ], dtype=np.complex128)
    SWAP = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.complex128)
    
# => Gate class <= #

class Gate(object):

    """ Class Gate represents a general N-body quantum gate. 

    An N-Body quantum gate applies a unitary operator to the state of a subset
    of N qubits, with an implicit identity matrix acting on the remaining
    qubits. The Gate class specifies the (2**N,)*2 unitary matrix U for the N
    active qubits, but does not specify which qubits are active.
    """

    def __init__(
        self,
        N,
        Ufun,
        params,
        name,
        ascii_symbols,
        ):

        """ Initializer. Params are set as object attributes.

        Params:
            N (int > 0) - the dimensionality of the quantum gate, e.g, 1 for
                1-body, 2 for 2-body, etc.
            Ufun (function of OrderedDict of str : float -> np.ndarray of shape
                (2**N,)*2) - a function which generates the unitary
                matrix for this gate from the current parameter set.
            params (OrderedDict of str : float) - the dictionary of initial
                gate parameters.
            name (str) - a simple name for the gate, e.g., 'CNOT'
            ascii_symbols (list of str of len N) - a list of ASCII symbols for
                each active qubit of the gate, for use in generating textual diagrams, e.g.,
                ['@', 'X'] for CNOT.
        """
        
        self.N = N
        self.Ufun = Ufun
        self.params = params
        self.name = name
        self.ascii_symbols = ascii_symbols

        # Validity checks
        if not isinstance(self.N, int): raise RuntimeError('N must be int')
        if self.N <= 0: raise RuntimeError('N <= 0') 
        if self.U.shape != (2**self.N,)*2: raise RuntimeError('U must be shape (2**N,)*2')
        if not isinstance(self.params, collections.OrderedDict): raise RuntimeError('params must be collections.OrderedDict')
        if not all(isinstance(_, str) for _ in self.params.keys()): raise RuntimeError('params keys must all be str')
        if not all(isinstance(_, float) for _ in self.params.values()): raise RuntimeError('params values must all be float')
        if not isinstance(self.name, str): raise RuntimeError('name must be str')
        if not isinstance(self.ascii_symbols, list): raise RuntimeError('ascii_symbols must be list')
        if len(self.ascii_symbols) != self.N: raise RuntimeError('len(ascii_symbols) != N')
        if not all(isinstance(_, str) for _ in self.ascii_symbols): raise RuntimeError('ascii_symbols must all be str')

    def __str__(self):
        """ String representation of this Gate (self.name) """
        return self.name
    
    @property
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
        return self.Ufun(self.params)

    # > Copying < #
    
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
            ascii_symbols=ascii_symbols,
            )

    # > Parameter Access < #

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

# > Explicit 1-body gates < #

""" I (identity) gate """
Gate.I = Gate(
    N=1,
    Ufun = lambda params : Matrix.I,
    params=collections.OrderedDict(),
    name='I',
    ascii_symbols=['I'],
    )
""" X (NOT) gate """
Gate.X = Gate(
    N=1,
    Ufun = lambda params : Matrix.X,
    params=collections.OrderedDict(),
    name='X',
    ascii_symbols=['X'],
    )
""" Y gate """
Gate.Y = Gate(
    N=1,
    Ufun = lambda params : Matrix.Y,
    params=collections.OrderedDict(),
    name='Y',
    ascii_symbols=['Y'],
    )
""" Z gate """
Gate.Z = Gate(
    N=1,
    Ufun = lambda params : Matrix.Z,
    params=collections.OrderedDict(),
    name='Z',
    ascii_symbols=['Z'],
    )
""" H (Hadamard) gate """
Gate.H = Gate(
    N=1,
    Ufun = lambda params : Matrix.H,
    params=collections.OrderedDict(),
    name='H',
    ascii_symbols=['H'],
    )
""" S gate """
Gate.S = Gate(
    N=1,
    Ufun = lambda params : Matrix.S,
    params=collections.OrderedDict(),
    name='S',
    ascii_symbols=['S'],
    )
""" T gate """
Gate.T = Gate(
    N=1,
    Ufun = lambda params : Matrix.T,
    name='T',
    params=collections.OrderedDict(),
    ascii_symbols=['T'],
    )
""" Rx2 gate """
Gate.Rx2 = Gate(
    N=1,
    Ufun = lambda params : Matrix.Rx2,
    params=collections.OrderedDict(),
    name='Rx2',
    ascii_symbols=['Rx2'],
    )
""" Rx2T gate """
Gate.Rx2T = Gate(
    N=1,
    Ufun = lambda params : Matrix.Rx2T,
    params=collections.OrderedDict(),
    name='Rx2T',
    ascii_symbols=['Rx2T'],
    )

# > Explicit 2-body gates < #

""" CNOT (CX) gate """
Gate.CNOT = Gate(
    N=2,
    Ufun = lambda params: Matrix.CX,
    params=collections.OrderedDict(),
    name='CNOT',
    ascii_symbols=['@', 'X'],
    )
Gate.CX = Gate.CNOT # Common alias
""" CY gate """
Gate.CY = Gate(
    N=2,
    Ufun = lambda params: Matrix.CY,
    params=collections.OrderedDict(),
    name='CY',
    ascii_symbols=['@', 'Y'],
    )
""" CZ gate """
Gate.CZ = Gate(
    N=2,
    Ufun = lambda params: Matrix.CZ,
    params=collections.OrderedDict(),
    name='CZ',
    ascii_symbols=['@', 'Z'],
    )
""" CS gate """
Gate.CS = Gate(
    N=2,
    Ufun = lambda params: Matrix.CS,
    params=collections.OrderedDict(),
    name='CS',
    ascii_symbols=['@', 'S'],
    )
""" SWAP gate """
Gate.CZ = Gate(
    N=2,
    Ufun = lambda params: Matrix.SWAP,
    params=collections.OrderedDict(),
    name='SWAP',
    ascii_symbols=['X', 'X'],
    )

# > Parametrized 1-body gates < #

@staticmethod
def _GateRx(theta):

    """ Rx (theta) = exp(-i * theta * x) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -1.j*s], [+1.j*s, c]], dtype=np.complex128)
    
    return Gate(
        N=1,
        Ufun=Ufun,
        params=collections.OrderedDict([('theta', theta)]),
        name='Rx',
        ascii_symbols=['Rx'],
        )
    
@staticmethod
def _GateRy(theta):

    """ Ry (theta) = exp(-i * theta * Y) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [+s, c]], dtype=np.complex128)

    return Gate(
        N=1,
        Ufun=Ufun,
        params=collections.OrderedDict([('theta', theta)]),
        name='Ry',
        ascii_symbols=['Ry'],
        )
    
@staticmethod
def _GateRz(theta):

    """ Rz (theta) = exp(-i * theta * Z) """
    
    def Ufun(params):
        theta = params['theta']
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c-1.j*s, 0.0], [0.0, c+1.j*s]], dtype=np.complex128)

    return Gate(
        N=1,
        Ufun=Ufun,
        params=collections.OrderedDict([('theta', theta)]),
        name='Rz',
        ascii_symbols=['Rz'],
        )
    
Gate.Rx = _GateRx
Gate.Ry = _GateRy
Gate.Rz = _GateRz

# > Parametrized 2-body gates < #

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

    return Gate(
        N=2,
        Ufun=Ufun,
        params=collections.OrderedDict([('A', A), ('B', B), ('C', C), ('D', D), ('E', E), ('F', F)]),
        name='SO4',
        ascii_symbols=['SO4A', 'SO4B'],
        )

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

    return Gate(
        N=2,
        Ufun=Ufun,
        params=collections.OrderedDict([
            ('thetaIY' , thetaIY),
            ('thetaYI' , thetaYI),
            ('thetaXY' , thetaXY),
            ('thetaYX' , thetaYX),
            ('thetaZY' , thetaZY),
            ('thetaYZ' , thetaYZ),
        ]),
        name='SO42',
        ascii_symbols=['SO42A', 'SO42B'],
        )

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
    
    return Gate(
        N=2,
        Ufun=Ufun,
        params=collections.OrderedDict([('theta', theta)]),
        name='CF',
        ascii_symbols=['@', 'F'],
        )

Gate.CF = _CF

# Special explicit gates 

@staticmethod
def _GateU1(U):

    """ An explicit 1-body gate that is specified by the user. """

    return Gate(
        N=1,
        Ufun = lambda params : U,
        params=collections.OrderedDict(),
        name='U1',
        ascii_symbols=['U1'],
        )

@staticmethod
def _GateU2(U):

    """ An explicit 2-body gate that is specified by the user. """

    return Gate(
        N=2,
        Ufun = lambda params : U,
        params=collections.OrderedDict(),
        name='U2',
        ascii_symbols=['U2A', 'U2B'],
        )

Gate.U1 = _GateU1
Gate.U2 = _GateU2

