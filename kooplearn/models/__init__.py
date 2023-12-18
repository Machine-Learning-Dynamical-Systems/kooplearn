from kooplearn.models.edmd import DMD, ExtendedDMD
from kooplearn.models.kernel import KernelDMD

from kooplearn.models.deepedmd import DeepEDMD  # isort:skip

try:
    from kooplearn._src.check_deps import check_torch_deps

    check_torch_deps()
    from kooplearn.models.ae.consistent import ConsistentAE
    from kooplearn.models.ae.dynamic import DynamicAE
except:
    pass
