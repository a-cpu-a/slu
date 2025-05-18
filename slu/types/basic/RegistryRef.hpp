/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slu/Include.hpp>

#include <slu/Utils.hpp>
#include <slu/types/Converter.hpp>

namespace slu
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
