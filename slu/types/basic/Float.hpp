/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slu/Include.hpp>

#include <slu/Utils.hpp>
#include <slu/types/Converter.hpp>

namespace slua
{
	struct Float
	{
		double val;

		constexpr Float() = default;
		constexpr Float(const double value) :val(value) {}

		static int push(lua_State* L, const Float& data)
		{
			lua_pushnumber(L, data.val);
			return 1;
		}
		static Float read(lua_State* L, const int idx) {
			return Float((double)lua_tonumber(L, idx));
		}
		static bool check(lua_State* L, const int idx) {
			return lua_isnumber(L, idx);
		}
		static constexpr const char* getName() { return LC_double; }
	};
}
// Map basic types to slua::Float, to allow easy pushing, reading, and checking
Slua_mapType(float, slua::Float);
Slua_mapType(double, slua::Float);