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
	struct Bool
	{
		bool val;

		Bool() {}
		Bool(const bool value) :val(value) {}

		static bool push(lua_State* L, const Bool& data)
		{
			lua_pushboolean(L, data.val);
			return true;
		}
		static Bool read(lua_State* L, const int idx) {
			return Bool(lua_toboolean(L, idx));
		}
		static bool check(lua_State* L, const int idx) {
			return lua_isboolean(L, idx);
		}
		static constexpr const char* getName() { return LUACC_BOOLEAN "bool" LUACC_DEFAULT; }
	};
}
// Map basic types to slua::Bool to allow easy pushing, reading, and checking
SLUA_MAP_TYPE(bool, slua::Bool);