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
	//does not allow C-functions
	//may, or may not be nil
	struct OptFunction
	{
		lua_State* L=nullptr;
		int idx = 0;

		constexpr OptFunction() = default;
		constexpr OptFunction(lua_State* _L, const int _idx) :L(_L), idx(_idx) {}


		static OptFunction read(lua_State* L, const int idx) {
			return OptFunction(L, idx);
		}
		static bool check(lua_State* L, const int idx)
		{	//allow lua functions, and nil, but not C-functions
			return (lua_isfunction(L, idx) || lua_isnil(L, idx)) && !lua_iscfunction(L, idx);
		}
		static constexpr const char* getName() { return LUACC_FUNCTION "optional-function" LUACC_DEFAULT; }
	};
}
