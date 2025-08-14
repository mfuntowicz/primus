#ifndef PRIMUS_TYPES
#define PRIMUS_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"

def Primus_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def Primus_FloatTensor : RankedTensorOf<[Primus_Float]>;

#endif // PRIMUS_TYPES