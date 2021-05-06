from .affine import affine
from .standard import standard
from .loop import loop
from .linalg import linalg
from .llvm import llvm

STANDARD_DIALECTS = [affine, standard, loop, linalg, llvm]
