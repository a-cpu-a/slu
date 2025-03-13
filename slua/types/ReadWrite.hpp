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


	template<slua::NonLuaType T>
	inline T read(lua_State* L, const int idx, T) {
		if constexpr (requires{slua::ToLua<T>::read(nullptr, 0).val; })
			return (T)slua::ToLua<T>::read(L, idx).val;
		else
			return slua::ToLua<T>::read(L, idx);
	}
	template<slua::LuaType T>
	inline T read(lua_State* L, const int idx, T) {
		return T::read(L, idx);
	}

	template<slua::NonLuaType T>
	inline T read(lua_State* L, const int idx) {
		if constexpr (requires{slua::ToLua<T>::read(nullptr, 0).val; })
			return (T)slua::ToLua<T>::read(L, idx).val;
		else
			return slua::ToLua<T>::read(L, idx);
	}
	template<slua::LuaType T>
	inline T read(lua_State* L, const int idx) {
		return T::read(L, idx);
	}


	/* Check if something on the lua stack is of a type, will throw with a message on some failures */
	template<typename T>
	inline bool checkThrowing(lua_State* L, const int idx, T) {
		return ToLua<T>::check(L, idx);
	}
	/* Check if something on the lua stack is of a type, will throw with a message on some failures */
	template<typename T>
	inline bool checkThrowing(lua_State* L, const int idx) {
		return ToLua<T>::check(L, idx);
	}


	/* Check if something on the lua stack is of a type */
	template<typename T>
	inline bool check(lua_State* L, const int idx, T) {
		try
		{
			return checkThrowing<T>(L, idx);
		}
		catch (...)
		{
			return false;//failed
		}
	}
	/* Check if something on the lua stack is of a type */
	template<typename T>
	inline bool check(lua_State* L, const int idx) {
		try
		{
			return checkThrowing<T>(L, idx);
		}
		catch (...)
		{
			return false;//failed
		}
	}

	template<typename T>
	inline constexpr const char* getName() {
		return ToLua<T>::getName();
	}
	template<typename T>
	inline constexpr const char* getName(T) {
		return ToLua<T>::getName();
	}
}