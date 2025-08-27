
add_subdirectory(parse)
add_subdirectory(ast)

add_library(slu 
	settings.ixx char_info.ixx big_int.ixx 
	lang/basic_state.ixx 
	lang/mpc.ixx 
	ext/mtx.ixx 
	"gen/output.ixx"
	SluBuildTest.cpp
)
