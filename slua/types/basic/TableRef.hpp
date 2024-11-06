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
	struct TableRef
	{
		lua_State* L;
		const int idx = 0;

		TableRef() {}
		TableRef(lua_State* _L, const int _idx) :L(_L), idx(_idx) {}

		//TODO: use [] operator instead
		void setU8(const int64_t i, const uint8_t value)
		{
			lua_pushinteger(L, value);
			lua_rawseti(L, idx, i);
		}
		size_t size() { return lua_rawlen(L, idx); }
		void push_back(const uint8_t value) {
			setU8(size() + 1, value);
		}

		static IntRef read(lua_State* L, const int idx) {
			return IntRef(L, idx);
		}
		static bool check(lua_State* L, const int idx) {
			return lua_istable(L, idx);
		}
		static constexpr const char* getName() { return  "table-reference"; }
	};
}
