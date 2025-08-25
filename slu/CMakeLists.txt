
add_subdirectory(parse)

add_library(slu 
	SluBuildTest.cpp settings.ixx char_info.ixx big_int.ixx 
	lang/basic_state.ixx 
	ext/mtx.ixx 
	"ast/pos.ixx"
	"ast/type.ixx"
	"ast/state.ixx"
	"ast/state_decls.ixx"
	"ast/op_order.ixx"
	"ast/small_enum_list.ixx"
	ast/enums.ixx 
	ast/op_info.ixx
)
