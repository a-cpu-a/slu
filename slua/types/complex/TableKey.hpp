/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slua/Include.hpp>

#include <slua/Utils.hpp>
#include <slua/types/Converter.hpp>
#include <slua/types/ReadWrite.hpp>
#include <slua/types/complex/String.hpp> //This uses slua::push, with a string

namespace slua
{
	enum class TableKeyType : uint8_t
	{
		INT,
		FLOAT,
		STRING
	};

	struct TableKey
	{
		std::string strVal;
		union
		{
			// one of these
			double floatVal;
			int64_t intVal;
		};

		TableKeyType type;

		std::string toString() const
		{
			switch (type)
			{
			case slua::TableKeyType::INT:
				return std::to_string(intVal);
			case slua::TableKeyType::FLOAT:
				return std::to_string(floatVal);
			case slua::TableKeyType::STRING:
				return strVal;
			}
			Slua_panic("Memory corrupted");
			std::abort();
		}

		static int push(lua_State* L, const TableKey& data)
		{
			switch (data.type)
			{
			case slua::TableKeyType::INT:
				lua_pushinteger(L, data.intVal);
				break;
			case slua::TableKeyType::FLOAT:
				lua_pushnumber(L, data.floatVal);
				break;
			case slua::TableKeyType::STRING:
				slua::push(L, data.strVal);
				break;
			}
			return 1;
		}
		static TableKey read(lua_State* L, const int idx) {

			TableKey ret;

			if (lua_isnumber(L, idx))
			{
				if (lua_isinteger(L, idx))
				{
					ret.intVal = lua_tointeger(L, idx);
					ret.type = TableKeyType::INT;
				}
				else
				{
					ret.floatVal = lua_tonumber(L, idx);
					ret.type = TableKeyType::FLOAT;
				}
			}
			else
			{
				ret.strVal = slua::readString(L, idx);
				ret.type = TableKeyType::STRING;
			}
			return ret;
		}
		static bool check(lua_State* L, const int idx) {
			return lua_isnumber(L, idx) || lua_isstring(L, idx);
		}
		static constexpr const char* getName() { return "table-key"; }

		constexpr bool operator==(const double& other) const {
			return type == TableKeyType::FLOAT && floatVal == other;
		}
		constexpr bool operator==(const int64_t& other) const {
			return type == TableKeyType::INT && intVal == other;
		}
		constexpr bool operator==(const std::string_view& other) const {
			return type == TableKeyType::STRING && strVal == other;
		}
		constexpr bool operator==(const char other[]) const {
			return type == TableKeyType::STRING && strVal == other;
		}
	};
}