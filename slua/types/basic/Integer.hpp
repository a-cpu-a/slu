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
		int64_t val;

		Int() {}
		Int(const int64_t value) :val(value) {}

		static int push(lua_State* L, const Int& data)
		{
			lua_pushinteger(L, data.val);
			return 1;
		}
		static Int read(lua_State* L, const int idx) {
			return Int((int64_t)lua_tointeger(L, idx));
		}
		static bool check(lua_State* L, const int idx) {
			return lua_isinteger(L, idx);
		}
		static constexpr const char* getName() { return LUACC_NUMBER "integer" LUACC_DEFAULT; }
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