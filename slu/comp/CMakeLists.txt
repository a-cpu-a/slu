add_library(slu-comp)
target_sources(slu-comp PUBLIC
  FILE_SET slu_modules TYPE CXX_MODULES
  FILES
	"conv_data.ixx"
	"handle_task.ixx"
	"comp_thread.ixx"
	"compile.ixx"
	"cfg.ixx"
	"mlir/conv.ixx"
 )

 
target_precompile_headers(slu-comp PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_LIST_DIR}/mlir/Pch.hpp>"
)