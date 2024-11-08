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
	struct Function
	{
		std::vector<uint8_t> func;
		std::string name;

		Function() {}

		static int push(lua_State* L, const Function& data)
		{
			if (data.func.empty())
			{
				lua_pushnil(L);
				return 1;
			}

			slua::loadFunction(L, data.func, data.name.c_str(), true);
			return 1;
		}
		static Function read(lua_State* L, const int idx) {
			if (lua_isnil(L, idx)) return Function();

			lua_pushvalue(L, idx);
			return slua::dumpFunction(L, true);
		}
		static bool check(lua_State* L, const int idx) {
			return (lua_isfunction(L, idx) || lua_isnil(L, idx)) && !lua_iscfunction(L, idx);
		}
		static constexpr const char* getName() { return LUACC_FUNCTION "lua-function" LUACC_DEFAULT; }
	};
}