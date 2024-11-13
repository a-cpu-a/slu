/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slua/Include.hpp>

namespace slua
{
	struct Context
	{
		lua_State* L = nullptr;
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
