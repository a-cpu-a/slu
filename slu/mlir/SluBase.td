
#ifndef SLU_BASE
#define SLU_BASE

include "mlir/IR/OpBase.td"

def Slu_Dialect : Dialect {
  let name = "slu";
  let cppNamespace = "::slu_dial";
}

#endif // SLU_BASE
