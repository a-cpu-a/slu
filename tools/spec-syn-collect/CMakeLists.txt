
add_executable(slu_spec_syn_collect)
init_target(slu_spec_syn_collect)


target_sources(slu_spec_syn_collect PUBLIC
  FILE_SET slu_tool_modules TYPE CXX_MODULES
  FILES
	"main.ixx"
)