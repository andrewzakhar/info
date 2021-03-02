
# Encoding - Decoding scheme
1. Create a model from (inheritance) `BaseModel` (src\csi\models.py)
```python
Class MyModel(BaseModel):
    def __init__(self):
        super(BaseModel, self).__init__()
        # DO Something
    
    def encode(self, x: np.ndarray):
        # DO something
        pass
    
    def decode(self, x: np.ndarray):
        # DO somthing
        pass
```
2. The input description can be found in `BaseModel` docstring
    1. Encoding are using default complex csi matrix with shape `30x50x64x2`
    2. Decoding can take any format but must return complex matrix with same shape a input for `encode`
3. Then in file under `model_testing\csi_simulation\encoding_decoding_scheme.py` dir, create a file like in `encoding_decoding_scheme.py`
and follow the code.
    1. If you won't commit anything you can place main file anywhere. 
