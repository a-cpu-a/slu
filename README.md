# Star Lua

Intended for C++ 20, might work for older versions.

.hpp for C++ header files, .h for C header files


```cpp

#include <slua/WrapFunc.hpp>
#include <slua/types/basic/Integer.hpp>  // Needed for automatic integer type encoding
#include <slua/types/complex/String.hpp> // Needed for automatic std::string encoding

inline slua::Ret<std::string> luaGet(const uint32_t& id) {
	
	if(id==UINT32_MAX)
		return slua::Error("Invalid id!");

	if(rand()<100)                     // This one automaticaly adds the function
		throw slua::Error("Unlucky!"); // name, and doesnt need slua::Ret<>
	
	return std::to_string(id);
}

inline luaL_Reg lib[] = 
{
	SLua_Wrap("get",luaGet),

	{NULL, NULL}
};

inline int initLib(lua_State* L)
{
	luaL_newlib(L, lib);
	return 1;
}
```