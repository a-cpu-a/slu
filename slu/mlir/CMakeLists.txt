
project ("slu-dialect")

list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
find_package(MLIR REQUIRED CONFIG)

include_directories(
  ${MLIR_DIR}/../../../../mlir/include
  ${MLIR_DIR}/../../../tools/mlir/include

  ${LLVM_DIR}/../../../../llvm/include
  ${LLVM_DIR}/../../../include
  ${CMAKE_CURRENT_LIST_DIR}/../../
  #${CMAKE_CURRENT_BINARY_DIR}/../../
  ${CMAKE_BINARY_DIR}
)

include(AddLLVM)
include(TableGen)
include(AddMLIR)

add_mlir_dialect(SluOps slu)
add_mlir_doc(SluDialect SluDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(SluOps SluOps Dialects/ -gen-op-doc -dialect slu)

add_mlir_dialect_library(slu-dialect
  SluDialect.cpp
  SluOps.cpp

  DEPENDS
  MLIRSluOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRUBDialect
  )
