/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slu/Include.hpp>
#include <glm/glm.hpp>

#include <slu/Utils.hpp>
#include <slu/types/Converter.hpp>
#include <slu/types/TypeUtils.hpp>

namespace slu
{
	template <typename T, bool INT, bool SIGNED, size_t BITS>
	struct Vec
	{
		T val;

		constexpr Vec() = default;
		constexpr Vec(const T& value) :val(value) {}

		static int push(lua_State* L, const Vec<T, INT, SIGNED, BITS>& data)
		{
			lua_newtable(L);
			for (int i = 0; i < T::length(); i++)
			{
				if constexpr (INT)
				{
					Slu_setTableValue(L, i + 1, lua_pushinteger(L, data.val[i]));
				}
				else
				{
					Slu_setTableValue(L, i + 1, lua_pushnumber(L, data.val[i]));
				}
			}
			return 1;
		}
		static Vec<T, INT, SIGNED, BITS> read(lua_State* L, const int idx)
		{

			Vec<T, INT, SIGNED, BITS> ret;

			for (int i = 0; i < T::length(); i++)
			{
				lua_rawgeti(L, idx, i + 1);
				if constexpr (INT)
					ret.val[i] = typename T::value_type(lua_tointeger(L, -1));
				else
					ret.val[i] = typename T::value_type(lua_tonumber(L, -1));
				lua_pop(L, 1);
			}

			return ret;
		}
		static bool check(lua_State* L, const int idx)
		{
			if (!lua_istable(L, idx))
				return false;
			for (int i = 0; i < T::length(); i++)
			{
				if (!slu::lua_isTableValueOfType(L, idx, i + 1, LUA_TNUMBER))
					return false;
			}
			return true;
		}
	private:
		static constexpr std::string getStrName()
		{
			char type;
			if (INT)
			{
				if (SIGNED)
					type = 'i'; //i64vec4
				else
					type = 'u'; //u64vec4
			}
			else
				type = 'f';//float, f64vec4

			return type + slu::cexpToString(BITS) + "vec" + slu::cexpToString(T::length()) + "\0";
		}
		Slu_wrapGetStrName(getStrName);
	};
}
// Map basic types to slu::Vec, to allow easy pushing, reading, and checking
Slu_mapType1(glm::u64vec4, slu::Vec<glm::u64vec4, true, false, 64>);
Slu_mapType1(glm::i64vec4, slu::Vec<glm::i64vec4, true, true, 64>);
Slu_mapType1(glm::u64vec3, slu::Vec<glm::u64vec3, true, false, 64>);
Slu_mapType1(glm::i64vec3, slu::Vec<glm::i64vec3, true, true, 64>);
Slu_mapType1(glm::u64vec2, slu::Vec<glm::u64vec2, true, false, 64>);
Slu_mapType1(glm::i64vec2, slu::Vec<glm::i64vec2, true, true, 64>);

Slu_mapType1(glm::u32vec4, slu::Vec<glm::u32vec4, true, false, 32>);
Slu_mapType1(glm::i32vec4, slu::Vec<glm::i32vec4, true, true, 32>);
Slu_mapType1(glm::u32vec3, slu::Vec<glm::u32vec3, true, false, 32>);
Slu_mapType1(glm::i32vec3, slu::Vec<glm::i32vec3, true, true, 32>);
Slu_mapType1(glm::u32vec2, slu::Vec<glm::u32vec2, true, false, 32>);
Slu_mapType1(glm::i32vec2, slu::Vec<glm::i32vec2, true, true, 32>);

Slu_mapType1(glm::u16vec4, slu::Vec<glm::u16vec4, true, false, 16>);
Slu_mapType1(glm::i16vec4, slu::Vec<glm::i16vec4, true, true, 16>);
Slu_mapType1(glm::u16vec3, slu::Vec<glm::u16vec3, true, false, 16>);
Slu_mapType1(glm::i16vec3, slu::Vec<glm::i16vec3, true, true, 16>);
Slu_mapType1(glm::u16vec2, slu::Vec<glm::u16vec2, true, false, 16>);
Slu_mapType1(glm::i16vec2, slu::Vec<glm::i16vec2, true, true, 16>);

Slu_mapType1(glm::vec2, slu::Vec<glm::vec2, false, true, 32>);
Slu_mapType1(glm::vec3, slu::Vec<glm::vec3, false, true, 32>);
Slu_mapType1(glm::vec4, slu::Vec<glm::vec4, false, true, 32>);

Slu_mapType1(glm::dvec2, slu::Vec<glm::dvec2, false, true, 64>);
Slu_mapType1(glm::dvec3, slu::Vec<glm::dvec3, false, true, 64>);
Slu_mapType1(glm::dvec4, slu::Vec<glm::dvec4, false, true, 64>);