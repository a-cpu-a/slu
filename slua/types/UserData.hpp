/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <array>
#include <type_traits>

#include <slua/Utils.hpp>

namespace slua
{
	inline const size_t TYPE_ID_SIZE = 2;

// After all typeids are initialized, this will contain a value that will always be more than any existing type id
	inline uint16_t _nextTypeId=1;

	template<typename TYPE>
	inline const uint16_t _typeId = _nextTypeId++;

	template<typename TYPE>
	inline uint16_t getTypeId() {
		return _typeId<TYPE>;
	}
	// Changes the last 2 bytes into the types id
	// Uses sizeof(PTR_T) to find where to change stuff
	template<typename TYPE,typename PTR_T>
	inline void setTypeIdSeperate(PTR_T* ptr)
	{
		uint8_t* dataPtr = reinterpret_cast<uint8_t*>(ptr);

		const uint16_t typeId = getTypeId<TYPE>();

		dataPtr[sizeof(PTR_T)+0] = typeId & 0xFF;
		dataPtr[sizeof(PTR_T)+1] = typeId >> 8;
	}
	// Changes the last 2 bytes into the types id
	// Uses sizeof(TYPE) to find where to change stuff
	template<typename TYPE>
	inline void setTypeId(TYPE* ptr) {
		setTypeIdSeperate<TYPE,TYPE>(ptr);
	}

	// First checks if it is a userdata, then
	// checks the last 2 bytes, returns true if all is good
	// Does NOT check lua_gettop
	template<typename TYPE>
	inline bool checkTypeId(lua_State* L, const int idx) 
	{
		if (!lua_isuserdata(L, idx))
			return false;

		const size_t len = lua_rawlen(L, idx);
		if (len <= 2)
			return false;//empty type, or too small for typeid

		const uint8_t* dataPtr = (const uint8_t*)lua_touserdata(L, idx);

		const uint16_t typeId = getTypeId<TYPE>();
		return (
			dataPtr[len-2] == (typeId & 0xFF))
			&& (
				dataPtr[len- 1] == (typeId >> 8));
	}


	// Use this inside __gc, to stop any double frees
	//
	// First checks if it is a userdata, then
	// checks the last 2 bytes, returns true if all is good
	// Does NOT check lua_gettop
	template<typename TYPE>
	inline bool checkAndClearTypeId(lua_State* L,const int idx) {
		if (!lua_isuserdata(L, idx))
			return false;

		const size_t len = lua_rawlen(L, idx);
		if (len <= 2)
			return false;//empty type, or too small for typeid


		uint8_t* dataPtr = reinterpret_cast<uint8_t*>(lua_touserdata(L,idx));

		const uint16_t typeId = getTypeId<TYPE>();

		if (
			dataPtr[len - 2] != (typeId & 0xFF)
			|| dataPtr[len - 1] != (typeId >>8)
			)
			return false;

		//clear it
		dataPtr[len - 2] = 0;
		dataPtr[len - 1] = 0;

		return true;
	}
}

//Pushes userdata
//Note: it will have 0 uservalues
template<class T,class... ARGS_T>
inline T& slua_newuserdata(lua_State* L,ARGS_T... constructorArgs)
{
	void* place = lua_newuserdatauv(L, sizeof(T) + slua::TYPE_ID_SIZE, 0);

	// https://isocpp.org/wiki/faq/dtors#placement-new
	auto id = new(place) T(constructorArgs...);
	slua::setTypeId(id);

	return *id;
}

template<class T>
inline int _slua_handleUserDataGC(lua_State* L)
{
	if (lua_gettop(L) != 1)
		return slua::lua_error(L, "__gc needs the original object!");
	if (!slua::checkAndClearTypeId<T>(L, 1))
		return slua::lua_error(L, "__gc needs a different type ... Mem Leak?");

	T* con = (T*)lua_touserdata(L, 1);

	con->~T();

	return 0;
}

template<class T>
inline bool slua_newMetatable(lua_State* L, const char* typeName)
{
	if (luaL_newmetatable(L, typeName))
	{
		if constexpr (!std::is_trivially_destructible_v<T>)
		{
			lua_pushcfunction(L, _slua_handleUserDataGC<T>);
			lua_setfield(L, -2, "__gc");
		}
		return true;
	}
	return false;
}