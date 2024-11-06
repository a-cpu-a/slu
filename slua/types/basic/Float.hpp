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
	struct Float
	{
		double val;

		Float() {}
		Float(const double value) :val(value) {}

		static bool push(lua_State* L, const Float& data)
		{
			lua_pushnumber(L, data.val);
			return true;
		}
		static Float read(lua_State* L, const int idx) {
			return Float((double)lua_tonumber(L, idx));
		}
		static bool check(lua_State* L, const int idx) {
			return lua_isnumber(L, idx);
		}
		static constexpr const char* getName() { return LUACC_NUMBER "double" LUACC_DEFAULT; }
	};
}
// Map basic types to slua::Float, to allow easy pushing, reading, and checking
SLUA_MAP_TYPE(float, slua::Float);
SLUA_MAP_TYPE(double, slua::Float);