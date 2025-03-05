# Sebastian Raschka 2014-2023
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np

from mlxtend._base import _BaseModel
from mlxtend.utils import assert_raises
import pytest
import time


class BlankModel(_BaseModel):
    def __init__(self, print_progress=0, random_seed=1):
        super().__init__()
        self.print_progress = print_progress
        self.random_seed = random_seed


def test_init():
    est = BlankModel(print_progress=0, random_seed=1)
    assert hasattr(est, "print_progress")
    assert hasattr(est, "random_seed")


def test_check_array_1():
    X = np.array([1, 2, 3])
    est = BlankModel()
    assert_raises(
        ValueError,
        "X must be a 2D array. Try X[:, numpy.newaxis]",
        est._check_arrays,
        X,
    )


def test_check_array_2():
    X = list([[1], [2], [3]])
    est = BlankModel(print_progress=0, random_seed=1)

    assert_raises(ValueError, "X must be a numpy array", est._check_arrays, X)


def test_check_array_3():
    X = np.array([[1], [2], [3]])
    est = BlankModel(print_progress=0, random_seed=1)
    est._check_arrays(X)

    
    
def test_get_params():
    """Test that get_params returns the expected parameter dictionary for BlankModel."""
    est = BlankModel(print_progress=5, random_seed=42)
    params = est.get_params()
    assert 'print_progress' in params
    assert 'random_seed' in params
    assert params['print_progress'] == 5
    assert params['random_seed'] == 42
    
    
def test_set_params():
    """Test that set_params updates the parameters in the estimator."""
    est = BlankModel(print_progress=0, random_seed=1)
    est.set_params(print_progress=10, random_seed=99)
    params = est.get_params()
    assert params['print_progress'] == 10
    assert params['random_seed'] == 99
    
    
# Dummy submodel for testing nested parameter access and update
class DummySubModel(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    def get_params(self, deep=True):
        return {'alpha': self.alpha}
    def set_params(self, **params):
        if 'alpha' in params:
            self.alpha = params['alpha']
        return self
    
    
# Dummy estimator to test nested parameters functionality in _BaseModel
class DummyEstimator(_BaseModel):
    def __init__(self, sub_model=None, gamma=1.0):
        self.sub_model = sub_model if sub_model is not None else DummySubModel()
        self.gamma = gamma
    
    
def test_get_and_set_nested_params():
    """Test that get_params nests parameters correctly and that set_params updates nested parameters."""
    est = DummyEstimator(gamma=1.0)
    params = est.get_params()
    # Expect nested key 'sub_model__alpha' along with 'gamma'
    assert 'gamma' in params
    assert 'sub_model__alpha' in params
    assert params['gamma'] == 1.0
    assert params['sub_model__alpha'] == 0.1
    
    # Now update nested parameter and gamma via set_params
    est.set_params(sub_model__alpha=0.5, gamma=2.0)
    params = est.get_params()
    assert params['gamma'] == 2.0
    assert params['sub_model__alpha'] == 0.5
    
    
def test_check_arrays_mismatched_y():
    """Test that _check_arrays raises ValueError when the number of samples in X and y do not match."""
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])  # mismatched length: 3 vs 2
    est = BlankModel(print_progress=0, random_seed=1)
    assert_raises(ValueError, "X and y must contain the same number of samples", est._check_arrays, X, y)
    
    
def test_bad_get_param_names():
    """Test that _get_param_names raises an error for models with variable positional arguments."""
    class BadModel(_BaseModel):
        def __init__(self, *args, param=0):
            pass
    with pytest.raises(RuntimeError, match="scikit-learn estimators should always specify their parameters"):
        _ = BadModel._get_param_names()
def test_init_time():
    """Test that _init_time attribute is set upon initialization with a valid timestamp."""
    est = BlankModel()
    # Verify that _init_time exists and is a recent timestamp (within the last 5 seconds)
    assert hasattr(est, "_init_time")
    assert time.time() - est._init_time < 5

def test_set_params_empty():
    """Test that calling set_params with no parameters does not change the estimator."""
    est = BlankModel(print_progress=3, random_seed=7)
    params_before = est.get_params()
    est.set_params()
    params_after = est.get_params()
    assert params_before == params_after

def test_check_arrays_y_invalid():
    """Test that passing a non-array y (without a shape attribute) raises a TypeError."""
    X = np.array([[1], [2], [3]])
    y = 5  # invalid y, not a numpy array
    est = BlankModel()
    with pytest.raises(TypeError):
        est._check_arrays(X, y)

def test_get_param_names_on_base():
    """Test that _get_param_names returns an empty list for _BaseModel since its __init__ takes only self."""
    param_names = _BaseModel._get_param_names()
    assert param_names == []

def test_invalid_param():
    """Test that setting an invalid parameter via set_params raises a ValueError."""
    est = BlankModel(print_progress=0, random_seed=1)
    with pytest.raises(ValueError, match="Invalid parameter"):
        est.set_params(nonexistent_param=123)

def test_get_params_deep_false():
    """Test that get_params with deep=False does not expand nested estimator parameters."""
    # Create a DummyEstimator to test nested parameter expansion
    est = DummyEstimator(gamma=3.0)
    params = est.get_params(deep=False)
    # The deep expansion should not be performed; thus no 'sub_model__alpha' key should be present,
    # although the 'sub_model' key (the nested object) should be.
    assert "sub_model__alpha" not in params
    assert "sub_model" in params
    assert params["gamma"] == 3.0
def test_get_param_names_custom():
    """Test that _get_param_names returns parameter names for a custom model."""
    class CustomModel(_BaseModel):
        def __init__(self, param1=1, param2=2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2

    expected = ['param1', 'param2']
    param_names = CustomModel._get_param_names()
    assert param_names == sorted(expected)

def test_set_params_non_estimator_nested():
    """Test that set_params works when a nested parameter does not have get_params."""
    class DummyNonEstimator:
        def __init__(self, val=100):
            self.val = val

    class ModelWithNonEstimator(_BaseModel):
        def __init__(self, dummy=None):
            super().__init__()
            self.dummy = dummy if dummy is not None else DummyNonEstimator()

    model = ModelWithNonEstimator()
    params = model.get_params()
    # There should be no nested keys since dummy doesn't implement get_params
    assert "dummy__val" not in params
    assert "dummy" in params

    # Now update dummy by passing a new DummyNonEstimator instance
    new_dummy = DummyNonEstimator(val=200)
    model.set_params(dummy=new_dummy)
    new_params = model.get_params()
    assert new_params["dummy"].val == 200

def test_set_nested_invalid_param_no_error():
    """Test that setting a nested parameter that does not exist in the submodel is silently ignored."""
    sub = DummySubModel(alpha=0.1)
    est = DummyEstimator(sub_model=sub, gamma=1.0)
    # Attempt to set a nested parameter that doesn't exist; the submodel should ignore it
    est.set_params(sub_model__nonexistent=123)
    params = est.get_params()
    # The valid parameter should remain unchanged
    assert params['sub_model__alpha'] == 0.1
def test_get_param_names_with_deprecated():
    """Test that _get_param_names retrieves parameters from a deprecated_original __init__ if available."""
    class DeprecatedModel(_BaseModel):
        def __init__(self, a=1, b=2):
            super().__init__()
            self.a = a
            self.b = b

    # Define a dummy deprecated __init__ with the intended signature.
    def dummy_init(self, a=1, b=2):
        super(DeprecatedModel, self).__init__()
        self.a = a
        self.b = b

    # Attach dummy_init as the deprecated_original.
    DeprecatedModel.__init__.deprecated_original = dummy_init

    param_names = DeprecatedModel._get_param_names()
    # Expected sorted list of parameters ['a', 'b']
    assert param_names == sorted(['a', 'b'])