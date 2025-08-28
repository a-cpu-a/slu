
target_sources(slu PUBLIC
  FILE_SET slu_modules TYPE CXX_MODULES
  FILES
	"conv_data.ixx"
	"handle_task.ixx"
	"cfg.ixx"
	"mlir/conv.ixx"
)