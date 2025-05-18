/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slu/Include.hpp>

#include <slu/Utils.hpp>
#include <slu/types/Converter.hpp>
#include <slu/types/ReadWrite.hpp>

namespace slua
{
	struct AnyRef
	{
		lua_State* L;
		int idx = 0;

		static int push(lua_State* L, const AnyRef& data)
		{
			lua_pushvalue(data.L, data.idx);//make copy
			return 1;
		}
		static AnyRef read(lua_State* L, const int idx) {
			return AnyRef(L, idx);
		}
		static bool check(lua_State* L, const int idx) {
			return true;
		}
		static constexpr const char* getName() { return "any"; }
	};
}