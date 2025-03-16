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
	struct RegistryRef
	{
		int refIdx = 0;

		constexpr RegistryRef() = default;
		constexpr RegistryRef(const int refIdx) :refIdx(refIdx) {}

		// Returns how many items were pushed to the stack, or negative in case of a error
		static int push(lua_State* L, const RegistryRef& data)
		{
			lua_rawgeti(L, LUA_REGISTRYINDEX, data.refIdx);
			return 1;
		}

		static constexpr const char* getName() { return  "registry-reference"; }
	};
}
