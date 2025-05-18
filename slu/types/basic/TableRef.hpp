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
	struct TableRef
	{
		int idx = 0;

		constexpr TableRef() = default;
		constexpr TableRef(const int _idx) :idx(_idx) {}

		//TODO: use [] operator instead
		void setU8(lua_State* L,const int64_t i, const uint8_t value)
		{
			lua_pushinteger(L, value);
			lua_rawseti(L, idx, i);
		}
		size_t size(lua_State* L) { return lua_rawlen(L, idx); }
		void push_back(lua_State* L, const uint8_t value) {
			setU8(L,size(L) + 1, value);
		}

		static TableRef read(lua_State* L, const int idx) {
			return TableRef(idx);
		}
		static bool check(lua_State* L, const int idx) {
			return lua_istable(L, idx);
		}
		static constexpr const char* getName() { return  "table-reference"; }
	};
}
