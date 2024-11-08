# Star Lua

Intended for C++ 20, might work for older versions.

.hpp for C++ header files, .h for C header files


```cpp
inline slua::Ret<std::string> luaGet(const uint32_t& id) {
	
	if(id==UINT32_MAX)
		return slua::Error("Invalid id!");

	if(rand()<100)
		throw slua::Error("Unlucky!"); //this one automaticaly adds the function name, and doesnt need slua::Ret<>
	
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