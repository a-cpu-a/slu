/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slua/Include.hpp>

#include <slua/Utils.hpp>
#include <slua/types/Converter.hpp>

namespace slua
{
	struct Int
	{
		int64_t val; // Special variable, used to unconvert automaticaly in SLua_MAP_TYPE

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
		static constexpr const char* getName() { return LC_integer; }
	};
}
// Map basic types to slua::Int, to allow easy pushing, reading, and checking
SLua_MAP_TYPE(uint8_t, slua::Int);
SLua_MAP_TYPE(uint16_t, slua::Int);
SLua_MAP_TYPE(uint32_t, slua::Int);
SLua_MAP_TYPE(uint64_t, slua::Int);
SLua_MAP_TYPE(int8_t, slua::Int);
SLua_MAP_TYPE(int16_t, slua::Int);
SLua_MAP_TYPE(int32_t, slua::Int);
SLua_MAP_TYPE(int64_t, slua::Int);