
target_sources(slu PRIVATE
  FILE_SET slu_modules TYPE CXX_MODULES
  FILES
	"sem_output.ixx"
	"to_html.ixx"
	"basic_col_conv.ixx"
	"paint_ops.ixx"
	"basics.ixx"
	"paint.ixx"
)
