
# Example test executable
add_executable(test_slu)
target_sources(test_slu PRIVATE
  FILE_SET test_slu_priv_modules TYPE CXX_MODULES
  FILES
	"test.ixx"
)
init_target(test_slu)
target_link_libraries(test_slu PRIVATE slu)
target_link_libraries(test_slu PRIVATE slu-comp)

# Register test with CTest
add_test(NAME SluTests COMMAND test_slu)