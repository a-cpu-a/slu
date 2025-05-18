/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <array>
#include <type_traits>

#include <slu/Utils.hpp>
#include <slu/types/ReadWrite.hpp>

namespace slua
{
	inline constexpr size_t TYPE_ID_SIZE = 2;

	// After all typeids are initialized, this will contain a value that will always be more than any existing type id
	inline uint16_t _nextTypeId = 1;

	template<typename TYPE>
	inline const uint16_t _typeId = _nextTypeId++;

	template<typename TYPE>
	inline uint16_t getTypeId() {
		return _typeId<TYPE>;
	}
	// Changes the last 2 bytes into the types id
	// Uses sizeof(PTR_T) to find where to change stuff
	template<typename TYPE, typename PTR_T>
	inline void setTypeIdSeperate(PTR_T* ptr)
	{
		uint8_t* dataPtr = reinterpret_cast<uint8_t*>(ptr);

		const uint16_t typeId = getTypeId<TYPE>();

		dataPtr[sizeof(PTR_T) + 0] = typeId & 0xFF;
		dataPtr[sizeof(PTR_T) + 1] = typeId >> 8;
	}
	// Changes the last 2 bytes into the types id
	// Uses sizeof(TYPE) to find where to change stuff
	template<typename TYPE>
	inline void setTypeId(TYPE* ptr) {
		setTypeIdSeperate<TYPE, TYPE>(ptr);
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
		if (len < TYPE_ID_SIZE)
			return false;//empty type, or too small for typeid

		const uint8_t* dataPtr = (const uint8_t*)lua_touserdata(L, idx);

		const uint16_t typeId = getTypeId<TYPE>();
		return (
			dataPtr[len - 2] == (typeId & 0xFF))
			&& (
				dataPtr[len - 1] == (typeId >> 8));
	}


	// Use this inside __gc, to stop any double frees
	//
	// First checks if it is a userdata, then
	// checks the last 2 bytes, returns true if all is good
	// Does NOT check lua_gettop
	template<typename TYPE>
	inline bool checkAndClearTypeId(lua_State* L, const int idx) {
		if (!lua_isuserdata(L, idx))
			return false;

		const size_t len = lua_rawlen(L, idx);
		if (len < TYPE_ID_SIZE)
			return false;//empty type, or too small for typeid


		uint8_t* dataPtr = reinterpret_cast<uint8_t*>(lua_touserdata(L, idx));

		const uint16_t typeId = getTypeId<TYPE>();

		if (
			dataPtr[len - 2] != (typeId & 0xFF)
			|| dataPtr[len - 1] != (typeId >> 8)
			)
			return false;

		//clear it
		dataPtr[len - 2] = 0;
		dataPtr[len - 1] = 0;

		return true;
	}

	template<class T>
	struct Userdata
	{
		lua_State* _L=nullptr;
		int _idx=0;//Position of the userdata on the stack

		static int push(lua_State* L, const Userdata<T>& data)
		{
			lua_pushvalue(data._L, data._idx);//make copy
			return 1;
		}
		static Userdata<T> read(lua_State* L, const int idx) {
			return Userdata<T>(L, idx);
		}
		static bool check(lua_State* L, const int idx) {
			return slua::checkThrowing<T>(L,idx);
		}
		static constexpr const char* getName() { return slua::getName<T>(); }

		T& get() const {
			return *((T*)lua_touserdata(_L, _idx));
		}

		T& operator*() const {
			return get();
		}
		T* operator->() const {
			return &get();
		}
	};
}

//Pushes userdata
//Note: it will have 0 uservalues
template<class T, class... ARGS_T>
inline T& slua_newUserdata(lua_State* L, ARGS_T&&... constructorArgs)
{
	//Should size be aligned to 2 bytes?
	void* place = lua_newuserdatauv(L, sizeof(T) + slua::TYPE_ID_SIZE, 0);

	// https://isocpp.org/wiki/faq/dtors#placement-new
	auto id = new(place) T(std::forward<ARGS_T>(constructorArgs)...);
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

#define _Slua_STRINGIZE(x) #x
#define _Slua_STRINGIZE2(x) _Slua_STRINGIZE(x)

// MUST NOT be inside a namespace !!!
// 
// 
//
#define _Slua_MAP_TYPE_2_USERDATA(_TY_NAME,_NAMESPACED_TYPE_ACCESS,_METATABLE_CUSTOMIZATION) \
	struct _sluaWrapperFor__ ## _TY_NAME { \
	static int push(lua_State* L, auto&& data) \
	{ \
		slua_newUserdata<_NAMESPACED_TYPE_ACCESS>(L,std::forward<decltype(data)>(data)); \
		if(slua_newMetatable<_NAMESPACED_TYPE_ACCESS>(L,#_TY_NAME  ":"  _Slua_STRINGIZE2(__LINE__))) \
			_METATABLE_CUSTOMIZATION;\
		lua_setmetatable(L, -2); \
		return 1; \
	} \
	static _NAMESPACED_TYPE_ACCESS& read(lua_State* L, const int idx) { \
		return *((_NAMESPACED_TYPE_ACCESS*)lua_touserdata(L,idx)); \
	} \
	static bool check(lua_State* L, const int idx) { \
		return slua::checkTypeId<_NAMESPACED_TYPE_ACCESS>(L, idx); \
	} \
	static constexpr const char* getName() { return _NAMESPACED_TYPE_ACCESS::getName(); } \
	}; \
	Slua_mapType(_NAMESPACED_TYPE_ACCESS,_sluaWrapperFor__ ## _TY_NAME)



// MUST NOT be inside a namespace !!!
// 
// _METATABLE_SETUP_FUNC -> a function taking 1 lua_State*, manipulating the table on the head of the stack.
//
#define Slua_mapType2Userdata(_TY_NAME,_NAMESPACED_TYPE_ACCESS,_METATABLE_SETUP_FUNC) \
	_Slua_MAP_TYPE_2_USERDATA(_TY_NAME,_NAMESPACED_TYPE_ACCESS,_METATABLE_SETUP_FUNC(L))

// MUST NOT be inside a namespace !!!
// 
// 
//
#define Slua_mapType2UserdataNoMetatable(_TY_NAME,_NAMESPACED_TYPE_ACCESS) \
	_Slua_MAP_TYPE_2_USERDATA(_TY_NAME,_NAMESPACED_TYPE_ACCESS,)