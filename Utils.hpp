/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>
#include <utility>
#include <bit>
#include <format>

#include <slua/Include.hpp>
#include <slua/ErrorType.hpp>

#if !defined(_MSC_VER) || defined(__clang__)
#define _Slua_NO_RETURN(...)
#else
#define _Slua_NO_RETURN(...) \
	do {__pragma(warning(push))__pragma(warning(suppress: 4645)) \
		return __VA_ARGS__; \
		__pragma(warning(pop)) \
	}while(0)
#endif

namespace slua
{
	//if a function is prefixed with lua_, its kinda unsafe

	using lua_Exception = struct lua_longjmp*;//Dont catch it, you ?might? break something.

	template<class... Args>
	[[noreturn]] inline int lua_error(lua_State* L, const std::format_string<Args...> fmt, Args&&... fmtArgs)
	{
		const std::string str = std::vformat(fmt.get(), std::make_format_args(fmtArgs...));

		lua_pushlstring(L, str.data(), str.size());

		lua_error(L);

		_Slua_NO_RETURN(0);
	}

	[[noreturn]] inline int lua_error(lua_State* L, const std::string& str) {
		lua_pushlstring(L, str.data(), str.size());
		lua_error(L);
		_Slua_NO_RETURN(0);
	}
	[[noreturn]] inline int lua_error(lua_State* L, const slua::Error& e) {
		lua_error(L, e.msg);
		_Slua_NO_RETURN(0);
	}
	[[noreturn]] inline int lua_error(lua_State* L, const char* str) {
		luaL_error(L, str);
		_Slua_NO_RETURN(0);
	}

	inline std::string readString(lua_State* L, const int idx) {
		size_t len;
		const char* str = lua_tolstring(L, idx, &len);

		return std::string(str, len);
	}

	// Sets the metatable of the top item on the stack to a global var (if that is a table)
	inline void lua_setGlobalMetatable(lua_State* L, const char* tableName)
	{
		if (lua_getglobal(L, tableName) == LUA_TTABLE)
			lua_setmetatable(L, -2);
		else
			lua_pop(L, 1);//not a table, so delete it
	}

	//Internal, used to load functions
	inline const char* _binExecuteReader(lua_State* L, void* data, size_t* size)
	{
		std::pair<const std::span<const uint8_t>*, bool>& binData = *((std::pair<const std::span<const uint8_t>*, bool>*)data);

		if (binData.second)
		{
			*size = 0;
			return NULL;
		}

		binData.second = true;

		*size = (*binData.first).size();
		return (const char*)((*binData.first).data());
	}
	inline int loadFunction(lua_State* L, const std::span<const uint8_t>& binData, const char* name, const bool binary = true)
	{
		_ASSERT(!binData.empty());

		std::pair pairData = std::pair(&binData, false);

		if (binary)
			return lua_load(L, _binExecuteReader, &pairData, name, "b");
		else
			return lua_load(L, _binExecuteReader, &pairData, name, "t");
	}
	inline int loadFunction(lua_State* L, const std::string& strData, const char* name, const bool binary = false) {
		return loadFunction(L,
			//convert from char to uint8_t
			std::bit_cast<std::span<const uint8_t>>(
				std::span<const char>(
					strData.begin(), strData.end()
				)
			),
			name, binary);
	}


	//Internal, used to dump functions
	inline int _binDumpWriter(lua_State* L, const void* data, size_t size, void* user)
	{
		std::vector<uint8_t>* vec = (std::vector<uint8_t>*)user;

		const uint8_t* dataPtr = (const uint8_t*)data;
		vec->insert(vec->end(), dataPtr, &dataPtr[size]);

		return 0;
	}
	inline std::vector<uint8_t> dumpFunction(lua_State* L, const bool pop = false, const bool noDebug = false)
	{
		std::vector<uint8_t> vec;

		lua_dump(L, _binDumpWriter, &vec, noDebug);

		if (pop)
			lua_pop(L, 1);

		return vec;
	}


	/*

	Table helpers

	*/


	//changes value of the table you just created, to the value you just created 
	inline void lua_setTableValue(lua_State* L, const lua_Integer idx) {
		lua_rawseti(L, -2, idx);
	}
	//returns the type of the element
	inline int lua_getTableValue(lua_State* L, const int idx, const lua_Integer key) {
		return lua_rawgeti(L, idx, key);
	}

	inline bool lua_isTableValueOfType(lua_State* L, const int idx, const lua_Integer key, const int type)
	{
		if (lua_rawgeti(L, idx, key) == type)
		{
			lua_pop(L, 1);
			return true;
		}
		lua_pop(L, 1);
		return false;
	}
}


//changes a value of the table you just created
#define Slua_setStrKeyTableValue(_L,_KEY,...) lua_pushstring(_L, (_KEY));__VA_ARGS__;lua_rawset(_L, -3)
#define Slua_setTableValue(_L,_IDX,...) __VA_ARGS__;slua::lua_setTableValue(_L,_IDX)