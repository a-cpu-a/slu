/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <optional>
#include <slu/Include.hpp>

#include <slu/Utils.hpp>
#include <slu/types/Converter.hpp>
#include <slu/types/ReadWrite.hpp>

namespace slu
{
	template<class T>
	struct Optional
	{
		std::optional<T> val; // Special variable, used to unconvert automaticaly in Slu_MAP_TYPE

		constexpr Optional() = default;
		constexpr Optional(const std::optional<T> value) :val(value) {}

		// Returns how many items were pushed to the stack, or negative in case of a error
		static int push(lua_State* L, const Optional<T>& data)
		{
			if (data.val.has_value())
				return slu::push(L, *data.val);

			lua_pushnil(L);
			return 1;
		}
		// Returns your type
		// And takes idx, which is the position of the item on the stack
		static Optional<T> read(lua_State* L, const int idx) {
			if (lua_isnil(L, idx))
				return {};
			return slu::read<T>(L,idx);
		}
		// Returns if succeded
		// And takes idx, which is the position of the item on the stack
		static bool check(lua_State* L, const int idx) {
			return lua_isnil(L, idx) || slu::checkThrowing<T>(L,idx);
		}
		// The name of your type, used inside error messages.
		static constexpr const char* getName() { return "optional"; }
	};

	template<typename T>
	struct __is_std_optional { using v = std::false_type; };
	template<typename T>
	struct __is_std_optional<std::optional<T>> { using v = std::true_type; };

	template<typename T>
	concept _is_std_optional = __is_std_optional<T>::v::value;
}
// Map basic types to slu::Optional, to allow easy pushing, reading, and checking
Slu_mapType(OPT_T, slu::Optional<typename OPT_T::value_type>, slu::_is_std_optional OPT_T);