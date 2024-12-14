/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slua/Include.hpp>

namespace slua
{
	struct Math
	{
		lua_State* L = nullptr;//Shared with the var inside Context

		double random() 
		{
			if (lua_getglobal(L, "math") != LUA_TTABLE)
				lua_pop(L, 1);
			else
			{
				lua_pushliteral(L, "random");
				if (lua_rawget(L, -2) != LUA_TFUNCTION
					|| !lua_iscfunction(L, -1))//require it to be a C function
					lua_pop(L, 2);//val & table
				else
				{
					if (lua_pcall(L, 0, 1, 0) == LUA_OK)
					{
						double ret = lua_tonumber(L, -1);
						lua_pop(L, 2);//val & table
						return ret;
					}
					lua_pop(L, 2);//val/err & table
				}
			}
			return 0.0;//error
		}
		//inclusive
		int64_t random(const int64_t min, const int64_t max)
		{
			if (lua_getglobal(L, "math") != LUA_TTABLE)
				lua_pop(L, 1);
			else
			{
				lua_pushliteral(L, "random");
				if (lua_rawget(L, -2) != LUA_TFUNCTION
					|| !lua_iscfunction(L, -1))//require it to be a C function
					lua_pop(L, 2);//val & table
				else
				{
					lua_pushinteger(L, min);
					lua_pushinteger(L, max);
					if (lua_pcall(L, 2, 1, 0) == LUA_OK)
					{
						int64_t ret = lua_tointeger(L, -1);
						lua_pop(L, 2);//val & table
						return ret;
					}
					lua_pop(L, 2);//val/err & table
				}
			}
			return min;//error
		}
	};

	struct Context
	{
		union //Both are just a lua_State*
		{
			lua_State* L = nullptr;
			Math math;//This will share the L variable
		};
		bool autoDelete = false;

		Context() {}
		Context(lua_State* L)
			: L(L)
		{}
		// Will automaticaly call lua_close
		Context(unsigned int rngSeed, lua_Alloc allocatorFunc = nullptr, void* allocatorUD = nullptr)
			: L(lua_newstate(allocatorFunc, allocatorUD, rngSeed)), autoDelete(true)
		{
			if (L == nullptr)
				throw std::bad_alloc();
		}
		~Context()
		{
			if (autoDelete)
				lua_close(L);
		}

	};
}
