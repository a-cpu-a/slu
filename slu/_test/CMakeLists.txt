add_library(_test-slu)
target_sources(_test-slu PRIVATE
  FILE_SET slu_priv_modules TYPE CXX_MODULES
  FILES
	"test.ixx"
)