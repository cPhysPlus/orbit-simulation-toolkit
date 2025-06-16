# Importing libraries
import pytest
from orbits import TwoBodyProblem
import numpy as np

class TestTwoBodyProblem:
    """Test suite for TwoBodyProblem class"""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures that are shared across all tests"""
        cls.valid_params = {
            'ecc': 0.5,
            'mass_bh': 1.0,
            'sm_axis': 1.0,
            'orb_period': 1.0,
            'method': 'RK3'
        }
    
    def setup_method(self):
        """Setup fresh test fixtures before each test"""
        self.problem = TwoBodyProblem(**self.valid_params)
    
    # Validation tests:

    def test_correct_input_values(self):
        """Verify constructor stores correct values"""
        assert self.problem.ecc == 0.5
        assert self.problem.mass_bh == 1.0
        assert self.problem.sm_axis == 1.0
        assert self.problem.orb_period == 1.0
        assert self.problem.method == 'RK3'
    
    def test_invalid_method(self):
        """Verify invalid methods raise ValueError"""
        invalid_params = self.valid_params.copy()
        invalid_params['method'] = 'InvalidMethod'
        
        with pytest.raises(ValueError) as excinfo:
            TwoBodyProblem(**invalid_params)
            
        # Check both the error type and message
        assert "Method not recognized" in str(excinfo.value)
        assert "Trapezoidal" in str(excinfo.value)  # Verify suggestions are included
    
    def test_different_inputs_different_outputs(self):
        """Verify different inputs produce different outputs"""
        # Create modified parameters
        alt_params = self.valid_params.copy()
        alt_params.update({
            'ecc': 0.8,
            'mass_bh': 2.0,
            'sm_axis': 1.5,
            'orb_period': 2.0
        })
        
        # Run simulations
        t1, s1 = self.problem.runge_kutta_3(t_span = [0, self.problem.period], dt = 100)
        problem2 = TwoBodyProblem(**alt_params)
        t2, s2 = problem2.runge_kutta_3(t_span = [0, problem2.period], dt = 100)
        
        # Verify outputs differ
        assert not np.array_equal(s1, s2)
    
    def teardown_method(self):
        """Clean up after each test"""
        del self.problem