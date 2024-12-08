'''
https://github.com/ryan-thompson/ContextualLasso.jl
'''
############## test 9 - test experiment using contextual_lasso ##############
import julia
from julia.api import Julia
jl = Julia(compiled_modules=False, runtime='/Users/amber/.juliaup/bin/julia')
jl.eval("""
ENV["JULIA_NUM_THREADS"] = 1
""")

import julia
from julia.api import Julia
j = Julia(compiled_modules=False, runtime='/Users/amber/.juliaup/bin/julia')
j.eval('using ChainRulesCore') 

j.eval('include("/Users/amber/Desktop/ece695_final_proj/contextual_lasso_example.jl")')



           