# CMakeList.txt : CMake project for slu-compiler-api, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("slu-compiler-api")

# Add source to this project's executable.
add_executable (slu-compiler-api "slu-compiler-api.cpp" "slu-compiler-api.h")

# add slu/
target_include_directories(slu-compiler-api
  PRIVATE
	${PROJECT_SOURCE_DIR}/../../
)

if(IS_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
endif()

find_package(MLIR REQUIRED CONFIG)

target_link_libraries(slu-compiler-api
    MLIRIR
    MLIRParser
    MLIRSupport

    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRIndexDialect
    
    MLIRIndexToLLVM
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRControlFlowInterfaces
    MLIRControlFlowToLLVM
    MLIRAffineTransforms
    MLIRAffineToStandard
    MLIRArithToLLVM
    MLIRArithTransforms
    MLIRFuncToLLVM
    MLIRFuncTransforms
    MLIRFunctionInterfaces
    MLIRMathToLLVM
    MLIRMathTransforms
    MLIRMemRefToLLVM
    MLIRMemRefTransforms

    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRPass
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRTransforms

    LLVMAArch64AsmParser
    LLVMAArch64CodeGen
    LLVMAMDGPUAsmParser
    LLVMAMDGPUCodeGen
    LLVMNVPTXCodeGen
    LLVMWebAssemblyAsmParser
    LLVMWebAssemblyCodeGen
    LLVMX86AsmParser
    LLVMX86CodeGen

    LLVMCodeGen
    LLVMTarget
    LLVMPasses
)

target_link_libraries(slu-compiler-api debug ${PROJECT_SOURCE_DIR}/../../x64/Debug/Slu.lib)
target_link_libraries(slu-compiler-api optimized ${PROJECT_SOURCE_DIR}/../../x64/Release/Slu.lib)

add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET slu-compiler-api PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
