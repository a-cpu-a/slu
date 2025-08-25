
add_subdirectory(parse)

add_library(slu 
	lang/mpc.ixx 
	SluBuildTest.cpp settings.ixx char_info.ixx big_int.ixx 
	lang/basic_state.ixx 
	ext/mtx.ixx 
	ast/enums.ixx ast/op_info.ixx)
