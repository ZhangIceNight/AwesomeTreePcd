#### Installation

```bash
# install hydra and omegaconf to use config yaml files
pip install hydra-core omegaconf
# to validate installation, please
python -c "import hydra; print(hydra.__version__)"
python -c "import omegaconf; print(omegaconf.OmegaConf.to_yaml({'key': 'value'}))"
# and it should be output
1.3.2
key: value

```

```bash
# set random seed
# you may get error like this
```
> RuntimeError: Deterministic behavior was enabled ... but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8
  
```python
# It is caused by torch.use_deterministic_algorithms(True)
# You can 
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
or os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
```


