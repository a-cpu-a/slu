
#ifndef SLU_DIAL
#define SLU_DIAL

include "mlir/IR/OpBase.td"

def Slu_Dialect : Dialect {
  let name = "slu";
  let cppNamespace = "::slu_dial";
}

class SluDialect_Op<string mnemonic, list<Trait> traits = []> :
    Op<Slu_Dialect, mnemonic, traits>;

#endif // SLU_DIAL
