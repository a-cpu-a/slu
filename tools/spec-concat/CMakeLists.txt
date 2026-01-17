
add_executable(slu_spec_concat)
init_target(slu_spec_concat)


target_sources(slu_spec_concat PUBLIC
  FILE_SET slu_tool_modules TYPE CXX_MODULES
  FILES
	"main.ixx"
)