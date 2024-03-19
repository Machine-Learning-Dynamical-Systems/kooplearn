from kooplearn.models.dict_of_fns import Linear, Nonlinear
from kooplearn.models.kernel import Kernel
from kooplearn.models.nystroem import NystroemKernel

try:
    from kooplearn._src.check_deps import check_torch_deps

    check_torch_deps()
    from kooplearn.models.ae.consistent import ConsistentAE
    from kooplearn.models.ae.dynamic import DynamicAE
except ImportError:
    pass
