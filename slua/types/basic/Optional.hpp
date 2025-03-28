/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <optional>
#include <slua/Include.hpp>

#include <slua/Utils.hpp>
#include <slua/types/Converter.hpp>
#include <slua/types/ReadWrite.hpp>

namespace slua
{
	template<class T>
	struct Optional
	{
		std::optional<T> val; // Special variable, used to unconvert automaticaly in SLua_MAP_TYPE

		constexpr Optional() = default;
		constexpr Optional(const std::optional<T> value) :val(value) {}

		// Returns how many items were pushed to the stack, or negative in case of a error
		static int push(lua_State* L, const Optional<T>& data)
		{
			if (data.val.has_value())
				return slua::push(L, *data.val);

			lua_pushnil(L);
			return 1;
		}
		// Returns your type
		// And takes idx, which is the position of the item on the stack
		static Optional<T> read(lua_State* L, const int idx) {
			if (lua_isnil(L, idx))
				return {};
			return slua::read<T>(L,idx);
		}
		// Returns if succeded
		// And takes idx, which is the position of the item on the stack
		static bool check(lua_State* L, const int idx) {
			return lua_isnil(L, idx) || slua::checkThrowing<T>(L,idx);
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
// Map basic types to slua::Optional, to allow easy pushing, reading, and checking
Slua_mapType(OPT_T, slua::Optional<typename OPT_T::value_type>, slua::_is_std_optional OPT_T);