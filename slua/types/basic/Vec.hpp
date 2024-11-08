/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slua/Include.hpp>
#include <glm/glm.hpp>

#include <slua/Utils.hpp>
#include <slua/types/Converter.hpp>

namespace slua
{
	template <typename T, bool INT, bool SIGNED, size_t BITS>
	struct Vec
	{
		T val;

		Vec() {}
		Vec(const T& value) :val(value) {}

		static int push(lua_State* L, const Vec<T, INT, SIGNED, BITS>& data)
		{
			lua_newtable(L);
			for (int i = 0; i < T::length(); i++)
			{
				if constexpr (INT)
					SLua_SetTableValue(L, i + 1, lua_pushinteger(L, data.val[i]));
				else
					SLua_SetTableValue(L, i + 1, lua_pushnumber(L, data.val[i]));
			}
			return 1;
		}
		static Vec<T, INT, SIGNED, BITS> read(lua_State* L, const int idx)
		{

			Vec<T, INT, SIGNED, BITS> ret;

			for (int i = 0; i < T::length(); i++)
			{
				lua_rawgeti(L, offset, i + 1);
				if constexpr (INT)
					ret.val[i] = T::value_type(lua_tointeger(L, -1));
				else
					ret.val[i] = T::value_type(lua_tonumber(L, -1));
				lua_pop(L, 1);
			}

			return ret;
		}
		static bool check(lua_State* L, const int idx)
		{
			if (!lua_istable(L, offset))
				return false;
			for (int i = 0; i < T::length(); i++)
			{
				if (!slua::lua_isTableValueOfType(L, offset, i + 1, LUA_TNUMBER))
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

			return ret + std::to_string(BITS) + "vec" + std::to_string(SIZE) + "\0";
		}
		inline const static constexpr std::array<char, getStrName().size()> name_buf = getStrName();
	public:

		static constexpr const char* getName() { return name_buf.data(); }
	};
}
// Map basic types to slua::Vec, to allow easy pushing, reading, and checking
SLua_MAP_TYPE(glm::u64vec4, slua::Vec<glm::u64vec4, true, false, 64>);
SLua_MAP_TYPE(glm::i64vec4, slua::Vec<glm::i64vec4, true, true, 64>);
SLua_MAP_TYPE(glm::u64vec3, slua::Vec<glm::u64vec3, true, false, 64>);
SLua_MAP_TYPE(glm::i64vec3, slua::Vec<glm::i64vec3, true, true, 64>);
SLua_MAP_TYPE(glm::u64vec2, slua::Vec<glm::u64vec2, true, false, 64>);
SLua_MAP_TYPE(glm::i64vec2, slua::Vec<glm::i64vec2, true, true, 64>);

SLua_MAP_TYPE(glm::u32vec4, slua::Vec<glm::u32vec4, true, false, 32>);
SLua_MAP_TYPE(glm::i32vec4, slua::Vec<glm::i32vec4, true, true, 32>);
SLua_MAP_TYPE(glm::u32vec3, slua::Vec<glm::u32vec3, true, false, 32>);
SLua_MAP_TYPE(glm::i32vec3, slua::Vec<glm::i32vec3, true, true, 32>);
SLua_MAP_TYPE(glm::u32vec2, slua::Vec<glm::u32vec2, true, false, 32>);
SLua_MAP_TYPE(glm::i32vec2, slua::Vec<glm::i32vec2, true, true, 32>);

SLua_MAP_TYPE(glm::u16vec4, slua::Vec<glm::u16vec4, true, false, 16>);
SLua_MAP_TYPE(glm::i16vec4, slua::Vec<glm::i16vec4, true, true, 16>);
SLua_MAP_TYPE(glm::u16vec3, slua::Vec<glm::u16vec3, true, false, 16>);
SLua_MAP_TYPE(glm::i16vec3, slua::Vec<glm::i16vec3, true, true, 16>);
SLua_MAP_TYPE(glm::u16vec2, slua::Vec<glm::u16vec2, true, false, 16>);
SLua_MAP_TYPE(glm::i16vec2, slua::Vec<glm::i16vec2, true, true, 16>);

SLua_MAP_TYPE(glm::vec2, slua::Vec<glm::vec2, false, true, 32>);
SLua_MAP_TYPE(glm::vec3, slua::Vec<glm::vec3, false, true, 32>);
SLua_MAP_TYPE(glm::vec4, slua::Vec<glm::vec4, false, true, 32>);

SLua_MAP_TYPE(glm::dvec2, slua::Vec<glm::dvec2, false, true, 64>);
SLua_MAP_TYPE(glm::dvec3, slua::Vec<glm::dvec3, false, true, 64>);
SLua_MAP_TYPE(glm::dvec4, slua::Vec<glm::dvec4, false, true, 64>);