import pandas as pd
from inspect import signature

class RedHawkPipe:
	def __init__(self, function):
		self.function = function
		self.numargs = len(signature(function).parameters)
		

	def __mul__(self, other):
		"""
		This defines vertical composition of pipes.
		"""


		def composition(x):
			p = other(x)
			return self(*p)
		return composition


	def __add__(self, other):
		"""
		This defines horizontal composition of pipes.
		"""


		def composition(x):
			nf = f.numargs
			ng = g.numargs
			lx = len(x)
			assert nf+ng == lx, \
			f'Expected {nf}+{ng}={nf+ng} arguments, got {lx}.'
			return (f(x[:num_args(f)]), g(x[:-num_args(g)]))
		return composition


	def decimate(self, parameters)

class RedHawkPipeline:
	def __init__(self, functions):
		self.input = input
		self.functions = functions

	def compose(self):
		return sum(functions)

	def run(self, input):
		function = self.compose()
		

