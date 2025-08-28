
target_sources(slu PRIVATE
  FILE_SET slu_modules TYPE CXX_MODULES
  FILES
	"make.ixx"
	"pos.ixx"
	"enums.ixx" 
	"op_info.ixx"
	"small_enum_list.ixx"
	"state_decls.ixx"
	"type.ixx"
	"state.ixx"
	"op_order.ixx"
	"mp_data.ixx" 
)