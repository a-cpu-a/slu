#include "slu/mlir/SluDialect.h"

using namespace mlir;
using namespace slu_dial;

#include "slu/mlir/SluOpsDialect.cpp.inc"

void slu_dial::SluDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "slu/mlir/SluOps.cpp.inc"
    >();
}