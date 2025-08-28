

add_library(slu)
target_sources(slu PUBLIC
  FILE_SET slu_modules TYPE CXX_MODULES
  FILES
	"settings.ixx"
	"char_info.ixx"
	"big_int.ixx"
	"ext/mtx.ixx"
	"num.ixx"
)
add_subdirectory(parse)
add_subdirectory(ast)
add_subdirectory(lang)
add_subdirectory(mlvl)
add_subdirectory(gen)
add_subdirectory(paint)

target_sources(slu PRIVATE
	SluBuildTest.cpp
)