# <img alt="Slua logo - hollow star with a cresent going through the middle" src="/slua/spec/info/Logo_white.png" width="120"> <img alt="Star Lua" src="/slua/spec/info/LogoText.svg" width="150"> 
Safe embedable scripting environments. 
 
For C++ 20.

# Function Wrapping
```cpp

#include <slua/WrapFunc.hpp>
#include <slua/types/basic/Integer.hpp>  // REQUIRED for automatic integer type encoding
#include <slua/types/complex/String.hpp> // REQUIRED for automatic std::string encoding

inline slua::Ret<std::string> luaGet(const uint32_t& id) {
	
	if( id == UINT32_MAX )
		return slua::Error("Invalid id!");

	if( rand() < 100 )                    // This one automaticaly adds the function name to
		throw slua::Error("Unlucky!"); // the message, and doesnt need slua::Ret<>

	if( id == 0 )
		return slua::Nil(); // This requires slua::Ret<>
	
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


# Custom Types
You can wrap types like in the example bellow, or
just add the methods to one of your
existing types (val wont be needed then).
```cpp

struct Int
{
	int64_t val; // Special variable name, used to unconvert automaticaly in SLua_MAP_TYPE

	Int() {}
	Int(const int64_t value) :val(value) {}

	// Returns how many items were pushed to the stack, or negative in case of a error
	static int push(lua_State* L, const Int& data)
	{
		lua_pushinteger(L, data.val);
		return 1;
	}
	// Returns your type
	// And takes idx, which is the position of the item on the stack
	static Int read(lua_State* L, const int idx) {
		return Int((int64_t)lua_tointeger(L, idx));
	}
	// Returns if succeded
	// And takes idx, which is the position of the item on the stack
	static bool check(lua_State* L, const int idx) {
		return lua_isinteger(L, idx);
	}
	// The name of your type, used inside error messages, so coloring is
	// a good idea (LUACC_NUMBER -> number color, LUACC_DEFAULT -> no color)
	static constexpr const char* getName() { return LUACC_NUMBER "integer" LUACC_DEFAULT; }
};

// Map uint8_t to slua::Int, to allow easy pushing, reading, and checking
SLua_MAP_TYPE(uint8_t, slua::Int);

```

# Lua Parsing
```cpp
std::vector<uint8_t> srcCode = ...;

sluaParse::Input in;
in.fName = ...;
in.text = srcCode; // "in.text" is a span, so no copy is made. "srcCode" needs to exist while "in" does.

sluaParse::ParsedFile f = sluaParse::parseFile(in);
```

# Style guide


.hpp for C++ header files, .h for old header files  

somethingLikeThis for variables, functions, namespaces.  
StartingWithUpperCase for types, concepts.  

prefixed with m_ for private things.  
prefixed with _ for implementation only stuff.  

Slua_nameNameName for macros.  
Slua_NameNameName for setting macros.  

_Slua_NAME_NAME_NAME for implementation macros.  
