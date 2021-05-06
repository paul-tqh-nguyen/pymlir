""" Implementation of the LLVM dialect. """


import inspect
import sys
from mlir.dialect import Dialect, DialectOp, DialectType, is_op, is_type

class LLVMVec(DialectType):
    _syntax_ = [
        ("llvm.vec < {size.integer_literal} x {type.type} >"),
        ("llvm.vec < ? x {size_factor.integer_literal} x {type.type} >"),
    ]

class LLVMPtr(DialectType):
    _syntax_ = "llvm.ptr< {type.type} >"

# Inspect current module to get all classes defined above
llvm = Dialect(
    "llvm",
    # TODO Add some operations
    # ops=[m[1] for m in inspect.getmembers(
    #     sys.modules[__name__], lambda obj: is_op(obj, __name__))],
    types=[m[1] for m in inspect.getmembers(
        sys.modules[__name__], lambda obj: is_type(obj, __name__))]
)
