/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <array>
#include <slua/Include.hpp>

#include <slua/Utils.hpp>
#include <slua/types/Converter.hpp>

namespace slua
{
	template<size_t SIZE>
	struct ByteArray
	{
		std::array<uint8_t, SIZE> val;

		ByteArray() {}
		ByteArray(const std::array<uint8_t, SIZE>& value) :val(value) {}

		static bool push(lua_State* L, const ByteArray& data)
		{
			lua_newtable(L);
			for (size_t i = 0; i < SIZE; i++)
			{
				lua_pushinteger(L, data.val[i]);
				slua::setTableValue(L, i + 1);
			}
			return true;
		}
		static ByteArray<SIZE> read(lua_State* L, const int idx) {
			ByteArray<SIZE> ret;

			const size_t checkLen = lua_rawlen(L, idx);

			for (size_t i = 0; i < checkLen; i++)//load in first ones
			{
				const int ty = getTableValue(L, idx, i + 1);
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
				const int ty = getTableElement(L, idx, i);
				lua_pop(L, 1);

				if (ty != LUA_TNUMBER)//|| ty == LUA_TSTRING
					return false;
			}
			return true;
		}

	private:
		static constexpr std::string getStrName() { return "byte-array[" LUACC_NUMBER + std::to_string(SIZE) + LUACC_DEFAULT "]\0"; }
		inline const static constexpr std::array<char, getStrName().size()> name_buf = getStrName();
	public:

		static constexpr const char* getName() { return name_buf.data(); }
	};
}
// Map basic types to slua::ByteArray to allow easy pushing, reading, and checking
SLUA_MAP_TYPE(std::array<uint8_t, SIZE>, slua::ByteArray<SIZE>, size_t SIZE);