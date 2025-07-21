# CMakeList.txt : CMake project for dialect, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)

project ("slu-dialect")

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")

include_directories(
  ${MLIR_DIR}/../../../../mlir/include
  ${MLIR_DIR}/../../../tools/mlir/include

  ${LLVM_DIR}/../../../../llvm/include
  ${LLVM_DIR}/../../../include
  ${CMAKE_CURRENT_LIST_DIR}/../../
)

include(AddLLVM)
include(TableGen)
include(AddMLIR)
#include(${MLIR_PATH}/AddMLIR.cmake)
#include(MLIR:AddMLIR)

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

  #mlir_tablegen(SluOps.h.inc -I ${MLIR_DIR}/../../../../mlir/include -I ${CMAKE_CURRENT_LIST_DIR}/../../ --gen-pass-decls)
  

#add_mlir_dialect_library(slu-dialect
#  MLIRFooOpsIncGen
#  MLIRFooTransformsIncGen
#)

#set(LLVM_TARGET_DEFINITIONS SluDialectOps.td)
#mlir_tablegen(SluDialectOps.h.inc -gen-op-decls)
#mlir_tablegen(SluDialectOps.cpp.inc -gen-op-defs)
#add_public_tablegen_target(SluDialectOpsIncGen)