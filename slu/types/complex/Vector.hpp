/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slu/Include.hpp>

#include <slu/Utils.hpp>
#include <slu/types/Converter.hpp>
#include <slu/types/ReadWrite.hpp>
#include <slu/types/TypeUtils.hpp>

namespace slu
{
	template<typename T, size_t MAX_VEC_SIZE = SIZE_MAX>
	struct Vector
	{
		std::vector<T> val;

		constexpr Vector() = default;
		template<size_t _S>
		constexpr Vector(const Vector<T, _S>& o)
			:val(o.val)
		{}
		constexpr Vector(const std::vector<T>& _v)
			:val(_v)
		{}

		static int push(lua_State* L, const Vector<T>& data)
		{
			lua_newtable(L);
			for (size_t i = 0; i < data.val.size(); i++)
			{
				slu::push(L, data.val[i]);
				lua_rawseti(L, -2, i + 1);
			}

			return 1;
		}
		static Vector<T> read(lua_State* L, const int idx)
		{
			Vector<T> ret;
			ret.val.reserve(lua_rawlen(L, idx));

			lua_pushnil(L);
			while (lua_next(L, idx) != 0)
			{
				const size_t i = lua_tointeger(L, -2);

				if (ret.val.size() < i)
					ret.val.resize(i);

				ret.val[i - 1] = slu::read<T>(L, lua_gettop(L));

				lua_pop(L, 1);//pop value
			}

			//lua_pop(L, 1);//pop key

			return ret;
		}
		static bool check(lua_State* L, int idx)
		{
			if (!lua_istable(L, idx))
				return false;

			size_t i = 0;

			lua_pushnil(L);
			while (lua_next(L, idx) != 0)
			{
				bool fail = i++ > MAX_VEC_SIZE || !slu::check<T>(L, lua_gettop(L)) || !lua_isinteger(L, -2);
				if (!fail)
				{
					const int64_t v = lua_tointeger(L, -2);
					fail |= (v - 1) > MAX_VEC_SIZE || v < 1;
				}
				if (fail)//or key isnt a integer
				{
					lua_pop(L, 2);//pop key & value
					return false;
				}
				lua_pop(L, 1);//pop value
			}

			return true;
		}


	private:
		static constexpr std::string getStrName()
		{
			const std::string parentName = slu::getName<T>();

			const std::string name = "array-of(" + parentName + ")";

			if (MAX_VEC_SIZE == SIZE_MAX)
				return name;// "unlimited"

			return name + "[" + slu::cexpToString(MAX_VEC_SIZE) + "]";
		}

		Slu_wrapGetStrName(getStrName);
	};

	template<typename T>
	struct __is_std_vector { using v = std::false_type; };
	template<typename T>
	struct __is_std_vector<std::vector<T>> { using v = std::true_type; };

	template<typename T>
	concept _is_std_vector = __is_std_vector<T>::v::value;
}
// Map basic types to slu::Vector, to allow easy pushing, reading, and checking
Slu_mapType(VEC_T, slu::Vector<typename VEC_T::value_type>, slu::_is_std_vector VEC_T);