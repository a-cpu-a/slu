/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <array>
#include <slua/Include.hpp>

#include <slua/Utils.hpp>
#include <slua/types/Converter.hpp>
#include <slua/types/TypeUtils.hpp>

namespace slua
{
	template<size_t SIZE>
	struct ByteArray
	{
		std::array<uint8_t, SIZE> val;

		ByteArray() {}
		ByteArray(const std::array<uint8_t, SIZE>& value) :val(value) {}

		static int push(lua_State* L, const ByteArray& data)
		{
			lua_newtable(L);
			for (size_t i = 0; i < SIZE; i++)
			{
				lua_pushinteger(L, data.val[i]);
				slua::lua_setTableValue(L, i + 1);
			}
			return 1;
		}
		static ByteArray<SIZE> read(lua_State* L, const int idx) {
			ByteArray<SIZE> ret;

			const size_t checkLen = lua_rawlen(L, idx);

			for (size_t i = 0; i < checkLen; i++)//load in first ones
			{
				const int ty = slua::lua_getTableValue(L, idx, i + 1);
				ret.val[i] = (uint8_t)lua_tointeger(L, -1);

				lua_pop(L, 1);
			}
			//set last ones to 0
			std::fill(ret.val.begin() + checkLen, ret.val.end(), 0);

			return ret;
		}
		static bool check(lua_State* L, const int idx)
		{
			if (!lua_istable(L, idx))
				return false;

			const size_t checkLen = lua_rawlen(L, idx);
			if (checkLen > SIZE)
				return false;//error, too long!

			for (size_t i = 0; i < checkLen; i++)//check types
			{
				const int ty = slua::lua_getTableValue(L, idx, i);
				lua_pop(L, 1);

				if (ty != LUA_TNUMBER)//|| ty == LUA_TSTRING
					return false;
			}
			return true;
		}

	private:
		static constexpr std::string getStrName() { return "byte-array[" LUACC_NUMBER + slua::cexpToString(SIZE) + LUACC_DEFAULT "]\0"; }

		SLua_WrapGetStrName(getStrName);
	};
}
// Map basic types to slua::ByteArray, to allow easy pushing, reading, and checking
SLua_MAP_TYPE(std::array<uint8_t SLua_CO SIZE>, slua::ByteArray<SIZE>, size_t SIZE);