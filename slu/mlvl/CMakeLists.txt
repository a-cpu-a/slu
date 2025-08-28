
target_sources(slu PUBLIC
  FILE_SET slu_modules TYPE CXX_MODULES
  FILES
	"resolve_type.ixx"
	"type_inf_check.ixx"
	"basic_desugar.ixx"
)