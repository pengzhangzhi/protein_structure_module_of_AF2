# Pytorch Implementation of Alphafold2's Structure Module 
![structure_module](figures/structure_module.png)
# Introduction

This is the Pytorch implementation of the structure module of Alphafold2 [1].
    
You can leverage this implementation to:

- understand the architecture of the structure module. Be sure also read the Alphafold2 supplement.
- incoorporate the structure module into your own model. All code is tested. 
- customize the structure module to your own needs. E.g., design a structure module for RNA structure prediction. See the examples.

# Install 



```bash
git clone https://github.com/pengzhangzhi/protein_structure_module_of_AF2.git
```
```bash
cd protein_structure_module_of_AF2
```

```python
pip install .
```


# Usage
start from 
https://github.com/pengzhangzhi/protein_structure_module_of_AF2/blob/b48e737bc5837fc64421a8fe5be52d9df6fbaeeb/structure_module/module.py#L25

# References

[1] Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021). https://doi.org/10.1038/s41586-021-03819-2

