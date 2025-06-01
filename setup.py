from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

quantum_lib = Extension(
    name="cquantum",
    sources=["cquantum.pyx"],
    language="c",
    libraries=["cquantum", "m"],
    library_dirs=["lib"],
    include_dirs=["lib"],
    extra_compile_args=["-mavx512vl", "-mavx512dq"],
    extra_link_args=[]
)

setup(
    name="cquantum",
    ext_modules=cythonize([quantum_lib])
)
