
#ifndef SLU_OPS
#define SLU_OPS

include "slu/mlir/SluDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

	
def SluDialect_ReinterpretMemRefOp : SluDialect_Op<"reinterpret_memref", [Pure]> {
  let summary = "Bit-level reinterpretation of memrefs with different element types and shapes";
  let description = [{
    Casts a memref to a different type, reinterpreting the same underlying memory buffer.
    Caller is responsible for ensuring the layout and size are compatible.
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

#endif // SLU_OPS
