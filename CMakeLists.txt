# CMakeList.txt : CMake project for slu, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("slu" LANGUAGES CXX)


# Add source to this project's executable.
add_subdirectory(slu)

set_property(TARGET slu PROPERTY CXX_STANDARD 20)

target_compile_options(slu PRIVATE "$<$<C_COMPILER_ID:MSVC>:/utf-8>")
target_compile_options(slu PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

target_include_directories(slu
  PRIVATE
	"${PROJECT_SOURCE_DIR}/slu/ext/llvm/mlir/include"
	"${PROJECT_SOURCE_DIR}/slu/ext/llvm/llvm/include"
	"${PROJECT_SOURCE_DIR}/slu/ext/llvm/build/tools/mlir/include"
	"${PROJECT_SOURCE_DIR}/slu/ext/llvm/build/include"
	"${PROJECT_SOURCE_DIR}/slu/ext/llvm/lld/include"
	"${PROJECT_SOURCE_DIR}/slu/ext/llvm/build/tools/lld/include"
	${PROJECT_SOURCE_DIR}
)

if (MSVC)
  target_compile_options(slu PRIVATE /bigobj)
endif ()

# TODO: Add tests and install targets if needed.
