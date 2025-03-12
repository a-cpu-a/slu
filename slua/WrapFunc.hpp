/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slua/types/Converter.hpp>
#include <slua/types/ReadWrite.hpp>
#include <slua/types/basic/Lua.hpp>
#include <slua/Context.hpp>

//include by default, cuz error messages are confusing
/*#include <slua/types/basic/Bool.hpp>
#include <slua/types/basic/Float.hpp>
#include <slua/types/basic/Integer.hpp>
#include <slua/types/basic/TableRef.hpp>
#include <slua/types/basic/Optional.hpp>

#include <slua/types/complex/ByteArray.hpp>
#include <slua/types/complex/Function.hpp>
#include <slua/types/complex/IntRef.hpp>
#include <slua/types/complex/OptFunc.hpp>
#include <slua/types/complex/String.hpp>
#include <slua/types/complex/TableKey.hpp>*/

namespace slua
{

	template<class RET_T, class... ARGS>
	using AnyCppFunc = RET_T(*)(ARGS...);

	/**
	* @throws slua::Error
	* @throws std::exception
	*/
	template <class RET_T,class... ARGS>
	inline int runCppFunc(lua_State* L, const std::string& funcName, AnyCppFunc<RET_T,ARGS...> func)
	{
		constexpr bool hasCtx = std::disjunction_v<std::is_same<slua::Context&,ARGS>...>;
		constexpr size_t argCount = sizeof...(ARGS) - (hasCtx ? 1 : 0);

		if (argCount != lua_gettop(L))
		{
			if constexpr (argCount == 1)
			{
				return slua::lua_error(L, LUACC_FUNCTION "Function "
					LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
					" needs " LUACC_NUMBER "1 " LC_argument ".");
			}
			return slua::lua_error(L, LUACC_FUNCTION "Function "
				LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
				" needs " LUACC_NUMBER + TS(sizeof...(ARGS)) + " " LC_arguments ".");
		}

		//TODO: replace the following with a template/constexpr for loop... or just reflection maybe
		std::tuple<std::remove_cvref_t<ARGS>...> args;

		int i = 0;
		std::apply(
			[&](auto&... argss) {(
				[&](auto& arg) {
					if constexpr (std::is_same_v<decltype(arg), slua::Context&>)//NOTE: reference to ctx
						arg= L;
					else
					{
						i++;
						if (slua::checkThrowing(L, i, arg))
							arg = slua::read(L, i, arg);
						else
							throw slua::Error(
								LUACC_ARGUMENT "Argument " 
								LUACC_NUMBER +std::to_string(i) + LUACC_DEFAULT 
								//" of " LUACC_FUNCTION "function " 
								//LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING 
								", is " LC_not " a "
								LUACC_END_SINGLE_STRING + slua::getName(arg) + LUACC_STRING_SINGLE "'"
							);
					}
				}(argss)
					, ...); }
		, args);

		if constexpr (std::is_same_v<RET_T, void>)
		{
			std::apply(func, args);
			return 0;// Nothing returned!
		}
		else
			return slua::push(L, std::apply(func, args));
	}


	inline int runCppFuncWrapped(lua_State* L, const std::string& funcName, const auto& func)
	{
		try
		{
			return runCppFunc(L, funcName, func);
		}
		catch (const std::exception& e)
		{
			return slua::lua_error(L, LC_Function " "
				LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
				" had a " LC_error ": "
				, e.what());
		}
		catch (const slua::Error& e)
		{
			return slua::lua_error(L, LC_Function " "
				LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
				" had a " LC_error ": "
				, e);
		}
	}



	// Wrap a C++ into a lua function, ment for lua_pushcfunction
#define Slua_wrapRaw(_LUA_NAME,_CPP_FUNC) [](lua_State* L){return ::slua::runCppFuncWrapped(L,_LUA_NAME,_CPP_FUNC);}

	// Wrap a C++ into a lua function, name pair, for your library function tables
#define Slua_wrap(_LUA_NAME,_CPP_FUNC) {_LUA_NAME,Slua_wrapRaw(_LUA_NAME,_CPP_FUNC)}




#define _Slua_MULTI_WRAP_CASE(_LUA_NAME,_COUNT,_CPP_FUNC) case _COUNT:return ::slua::runCppFuncWrapped(L,_LUA_NAME "(" #_COUNT ")",_CPP_FUNC)


	// Wrap 2 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
#define Slua_wrap2Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2) [](lua_State* L){switch(lua_gettop(L)){ \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		default: return ::slua::lua_error(L,LUACC_FUNCTION "Function " \
			LUACC_SINGLE_STRING( _LUA_NAME) \
			" needs " LUACC_NUMBER #_C1 LUACC_DEFAULT " or " LUACC_NUMBER #_C2 " " LC_arguments "." );} }

	// Wrap 2 C++ functions overloaded by arg count into 1 lua function, name pair, for your library function tables
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
#define Slua_wrap2(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2) {_LUA_NAME,Slua_wrap2Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2)}


	// Wrap 3 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
#define Slua_wrap3Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3) [](lua_State* L){switch(lua_gettop(L)){ \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C3,_CPP_FUNC3); \
		default: return ::slua::lua_error(L,LUACC_FUNCTION "Function " \
			LUACC_SINGLE_STRING( _LUA_NAME) \
			" needs " LUACC_NUMBER #_C1 LUACC_DEFAULT \
			", " LUACC_NUMBER #_C2 LUACC_DEFAULT \
			" or " LUACC_NUMBER #_C3 " " LC_arguments "." );} }

	// Wrap 3 C++ functions overloaded by arg count into 1 lua function, name pair, for your library function tables
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
#define Slua_wrap3(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3) {_LUA_NAME,Slua_wrap3Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3)}


	// Wrap 4 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
	// _C4 is the argument count of _CPP_FUNC4
#define Slua_wrap4Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4) [](lua_State* L){switch(lua_gettop(L)){ \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C3,_CPP_FUNC3); \
		_Slua_MULTI_WRAP_CASE(_LUA_NAME,_C4,_CPP_FUNC4); \
		default: return ::slua::lua_error(L,LUACC_FUNCTION "Function " \
			LUACC_SINGLE_STRING( _LUA_NAME) \
			" needs " LUACC_NUMBER #_C1 LUACC_DEFAULT \
			", " LUACC_NUMBER #_C2 LUACC_DEFAULT \
			", " LUACC_NUMBER #_C3 LUACC_DEFAULT \
			" or " LUACC_NUMBER #_C4 " " LC_arguments "." );} }

	// Wrap 4 C++ functions overloaded by arg count into 1 lua function, name pair, for your library function tables
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
	// _C4 is the argument count of _CPP_FUNC4
#define Slua_wrap4(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4) {_LUA_NAME,Slua_wrap4Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4)}
}