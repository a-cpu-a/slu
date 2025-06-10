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

#set_property(TARGET slu-compiler-api PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(slu-compiler-api debug ${PROJECT_SOURCE_DIR}/../../x64/Debug/Slu.lib)
target_link_libraries(slu-compiler-api optimized ${PROJECT_SOURCE_DIR}/../../x64/Release/Slu.lib)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET slu-compiler-api PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
