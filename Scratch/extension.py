from fenics import *
import instant
import os

cpp_src_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

header_file = open(cpp_src_dir + "AdexPointIntegralSolver.h", "r")
code = header_file.read()
header_file.close()

# system_headers = [
#     'numpy/arrayobject.h',
#     'dolfin/geometry/BoundingBoxTree.h',
#     'dolfin/function/FunctionSpace.h'
# ]

# cmake_packages = ['DOLFIN']
sources        = ["AdexPointIntegralSolver.cpp"]
source_dir     = cpp_src_dir
include_dirs   = [".", cpp_src_dir, '/usr/lib/petscdir/3.7.3/x86_64-linux-gnu-real/include/']
module_name    = "test"

# compile this with Instant JIT compiler :
inst_params = {
    'code'                      : code,
    # 'module_name'               : module_name,
    'source_directory'          : cpp_src_dir,
    'sources'                   : sources,
    # 'additional_system_headers' : ["petscsys.h"],
    # 'include_dirs'              : include_dirs
}
compiled_module = compile_extension_module(**inst_params)
