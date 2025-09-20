# CMakeList.txt : CMake project for slu, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18)

# Enable Hot Reload for MSVC compilers if supported.
if (MSVC)
	if (POLICY CMP0141)
	  cmake_policy(SET CMP0141 NEW)
	  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
	endif()
endif()

project ("slu" LANGUAGES CXX)


# Add source to this project's executable.
add_subdirectory(slu)

function(init_target target)
	set_property(TARGET ${target} PROPERTY CXX_STANDARD 23)
    target_compile_options(${target} PRIVATE
        "$<$<C_COMPILER_ID:MSVC>:/utf-8>"
        "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
    )
	target_include_directories(${target}
	  PRIVATE
		${PROJECT_SOURCE_DIR}
	)
	
	target_include_directories(${target}
	  PRIVATE
		"${PROJECT_SOURCE_DIR}/slu/ext/llvm/mlir/include"
		"${PROJECT_SOURCE_DIR}/slu/ext/llvm/llvm/include"
		"${PROJECT_SOURCE_DIR}/slu/ext/llvm/build/tools/mlir/include"
		"${PROJECT_SOURCE_DIR}/slu/ext/llvm/build/include"
		"${PROJECT_SOURCE_DIR}/slu/ext/llvm/lld/include"
		"${PROJECT_SOURCE_DIR}/slu/ext/llvm/build/tools/lld/include"
	)
endfunction()

init_target(slu)
init_target(slu-comp)
init_target(_test-slu)

target_link_libraries(slu-comp slu)
target_link_libraries(_test-slu slu-comp)
target_link_libraries(_test-slu slu)

if (MSVC)
  target_compile_options(slu PRIVATE /bigobj)
endif ()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# TODO: Add tests and install targets if needed.
