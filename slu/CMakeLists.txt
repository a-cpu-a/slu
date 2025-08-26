
add_subdirectory(parse)

add_library(slu 
	settings.ixx char_info.ixx big_int.ixx 
	lang/basic_state.ixx 
	lang/mpc.ixx 
	ext/mtx.ixx 
	"ast/pos.ixx"
	"ast/enums.ixx" 
	"ast/op_info.ixx"
	"ast/small_enum_list.ixx"
	"ast/state_decls.ixx"
	"ast/type.ixx"
	"ast/state.ixx"
	"ast/op_order.ixx"
	"ast/mp_data.ixx"
	SluBuildTest.cpp
)
