# Star Lua

Intended for C++ 20, might work for older versions.

.hpp for C++ header files, .h for C header files


```cpp
inline slua::Ret<std::string> luaGet(const uint32_t& id) {
	return std::to_string(id);
}

inline static luaL_Reg lib[] = 
{
	SLua_Wrap("get",luaGet),

	{NULL, NULL}
};

static int initLib(lua_State* L)
{
	luaL_newlib(L, lib);
	return 1;
}
```