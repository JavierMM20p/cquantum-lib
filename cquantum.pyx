cdef extern from "immintrin.h":
    ctypedef float __m512
    ctypedef float __m256
    ctypedef int __m256i
    ctypedef int __m512i
    __m512 _mm512_mul_ps(__m512, __m512)
    __m512 _mm512_add_ps(__m512, __m512)
    __m512 _mm512_permutexvar_ps(__m512i, __m512)
    __m512i _mm512_set_epi32(...)  # if used
    __m256 _mm512_castps512_ps256(__m512)
    __m256 _mm512_extractf32x8_ps(__m512, const int)
    __m256 _mm256_hadd_ps(__m256, __m256)
    __m512 _mm512_insertf32x8(__m512, __m256, const int)
    __m512 _mm512_castps256_ps512(__m256)

cdef extern from "state_vector.h":

    ctypedef struct StateVector:
        __m512* state_vec
        int n_qubits
        int n_vectors
        
    StateVector init_state_vector(int qubits)
    void state_vector_to_string(StateVector* state_vect)
    void state_vector_probability_to_string(StateVector* state_vect)
    void free_state_vector(StateVector* state_vect)
    void state_vector_polar_to_string(StateVector* state_vect)

cdef extern from "qgates.h":

    void debug_gate(StateVector* state_vect, int qubit)
    void debug_gate_d(StateVector* state_vect, int c_qubit, int qubit)

    void X_gate(StateVector* state_vect, int qubit)
    void Y_gate(StateVector* state_vect, int qubit)
    void Z_gate(StateVector* state_vect, int qubit)
    void H_gate(StateVector* state_vect, int qubit)
    void S_gate(StateVector* state_vect, int qubit)
    void T_gate(StateVector* state_vect, int qubit)
    void CNOT_gate(StateVector* state_vect, int control_qubit, int qubit)
    void SWAP_gate(StateVector* state_vect, int control_qubit, int qubit)
    void C_Phase_gate(StateVector* state_vect, int control_qubit, int qubit, float phi)

    int Z_measure(StateVector* state_vect, int qubit)

cdef extern from "circuits.h":
    void general_QFT(StateVector* state_vect)

cdef class PyStateVector:
    cdef StateVector sv
    def __cinit__(self, int qubits):
        self.sv = init_state_vector(qubits)
    
    def to_string(self):
        state_vector_to_string(&self.sv)
    
    def probability_to_string(self):
        state_vector_probability_to_string(&self.sv)
    
    def polar_to_string(self):
        state_vector_polar_to_string(&self.sv)
    
    def apply_X_gate(self, int qubit):
        X_gate(&self.sv, qubit)
    
    def apply_Y_gate(self, int qubit):
        Y_gate(&self.sv, qubit)
    
    def apply_Z_gate(self, int qubit):
        Z_gate(&self.sv, qubit)
    
    def apply_H_gate(self, int qubit):
        H_gate(&self.sv, qubit)
    
    def apply_S_gate(self, int qubit):
        S_gate(&self.sv, qubit)
    
    def apply_T_gate(self, int qubit):
        T_gate(&self.sv, qubit)
    
    def apply_CNOT_gate(self, int control_qubit, int qubit):
        CNOT_gate(&self.sv, control_qubit, qubit)
    
    def apply_SWAP_gate(self, int control_qubit, int qubit):
        SWAP_gate(&self.sv, control_qubit, qubit)
    
    def apply_C_Phase_gate(self, int control_qubit, int qubit, float phi):
        C_Phase_gate(&self.sv, control_qubit, qubit, phi)
    
    def debug_gate(self, int qubit):
        debug_gate(&self.sv, qubit)
    
    def debug_gate_d(self, int c_qubit, int qubit):
        debug_gate_d(&self.sv, c_qubit, qubit)
    
    def measure_Z(self, int qubit):
        return Z_measure(&self.sv, qubit)
    
    def apply_general_QFT(self):
        general_QFT(&self.sv)
