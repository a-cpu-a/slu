
#ifndef MLIR_DIALECT_SLU_H_
#define MLIR_DIALECT_SLU_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// Slu Dialect
//===----------------------------------------------------------------------===//

#include "slu/mlir/SluOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Slu Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "slu/mlir/SluOps.h.inc"

#endif // MLIR_DIALECT_SLU_H_