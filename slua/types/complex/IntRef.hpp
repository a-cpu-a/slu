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
	struct IntRef
	{
		lua_State* L=nullptr;
		int idx = 0;

		constexpr IntRef() = default;
		IntRef(lua_State* _L, const int _idx) :L(_L), idx(_idx) {}


		int64_t get()
		{
			lua_rawgeti(L, idx, 1);

			if (!lua_isinteger(L, -1))
			{
				return INT64_MIN;// uhhh, thats not right
				lua_pop(L, 1);
			}

			const int64_t ret = lua_tointeger(L, -1);
			lua_pop(L, 1);
			return ret;
		}
		IntRef& operator=(const int64_t value)
		{
			lua_pushinteger(L, value);
			lua_rawseti(L, idx, 1);
			return *this; // Return a reference to this object
		}


		//var++ operator
		int64_t operator ++(int) {
			const uint64_t ret = get();
			*this = ret + 1;
			return ret;
		}

		static IntRef read(lua_State* L, const int idx) {
			return IntRef(L, idx);
		}
		static bool check(lua_State* L, const int idx) {
			return lua_istable(L, idx) && slua::lua_isTableValueOfType(L, idx, 1, LUA_TNUMBER);
		}
		static constexpr const char* getName() { return LUACC_NUMBER "integer-reference" LUACC_DEFAULT; }
	};
}
