/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slua/Include.hpp>

#include "Converter.hpp"

namespace slua
{
	/* Push something to the lua stack */
	template <typename T>
	inline int push(lua_State* L, const T& data) {

		using LuaType = slua::ToLua<T>;

		return LuaType::push(L, LuaType(data));
	}


	template<typename T>
	inline T read(lua_State* L, const int idx, T) {
		return (T)slua::ToLua<T>::read(L, idx).val;
	}
	template<slua::NonLuaType T>
	inline T read(lua_State* L, const int idx) {
		return (T)slua::ToLua<T>::read(L, idx).val;
	}
	template<slua::LuaType T>
	inline T read(lua_State* L, const int idx) {
		return T::read(L, idx);
	}


	/* Check if something on the lua stack is of a type */
	template<typename T>
	inline bool check(lua_State* L, const int idx, T) {
		return ToLua<T>::check(L, idx);
	}
	/* Check if something on the lua stack is of a type */
	template<typename T>
	inline bool check(lua_State* L, const int idx) {
		return ToLua<T>::check(L, idx);
	}


	template<typename T>
	inline constexpr const char* getName() {
		return ToLua<T>::getName();
	}
}