
add_subdirectory(parse)
add_subdirectory(ast)
add_subdirectory(lang)
add_subdirectory(mlvl)
add_subdirectory(gen)
add_subdirectory(paint)

add_library(slu 
	settings.ixx
	char_info.ixx
	big_int.ixx
	ext/mtx.ixx
	SluBuildTest.cpp
 "num.ixx")
