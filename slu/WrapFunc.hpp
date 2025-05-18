/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/types/Converter.hpp>
#include <slu/types/ReadWrite.hpp>
#include <slu/types/basic/Lua.hpp>
#include <slu/Context.hpp>

//include by default, cuz error messages are confusing
/*#include <slu/types/basic/Bool.hpp>
#include <slu/types/basic/Float.hpp>
#include <slu/types/basic/Integer.hpp>
#include <slu/types/basic/TableRef.hpp>
#include <slu/types/basic/Optional.hpp>

#include <slu/types/complex/ByteArray.hpp>
#include <slu/types/complex/Function.hpp>
#include <slu/types/complex/IntRef.hpp>
#include <slu/types/complex/OptFunc.hpp>
#include <slu/types/complex/String.hpp>
#include <slu/types/complex/TableKey.hpp>*/

namespace slu
{

	template<class RET_T, class... ARGS>
	using AnyCppFunc = RET_T(*)(ARGS...);

	/**
	* @throws slu::Error
	* @throws std::exception
	*/
	template <class RET_T,class... ARGS>
	inline int runCppFunc(lua_State* L, const std::string& funcName, AnyCppFunc<RET_T,ARGS...> func)
	{
		constexpr bool hasCtx = std::disjunction_v<std::is_same<slu::Context&,ARGS>...>;
		constexpr size_t argCount = sizeof...(ARGS) - (hasCtx ? 1 : 0);

		if (argCount != lua_gettop(L))
		{
			if constexpr (argCount == 1)
			{
				return slu::lua_error(L, LUACC_FUNCTION "Function "
					LUACC_SINGLE_STRING("{}")
					" needs " LUACC_NUMBER "1 " LC_argument ".", funcName);
			}
			return slu::lua_error(L, LUACC_FUNCTION "Function "
				LUACC_SINGLE_STRING("{}")
				" needs " LUACC_NUMBER "{} " LC_arguments ".", funcName, sizeof...(ARGS));
		}

		//TODO: replace the following with a template/constexpr for loop... or just reflection maybe
		std::tuple<std::remove_cvref_t<ARGS>...> args;

		int i = 0;
		std::apply(
			[&](auto&... argss) {(
				[&](auto& arg) {
					if constexpr (std::is_same_v<decltype(arg), slu::Context&>)//NOTE: reference to ctx
						arg= L;
					else
					{
						i++;
						if (slu::checkThrowing(L, i, arg))
							arg = slu::read(L, i, arg);
						else
							throw slu::Error(
								LUACC_ARGUMENT "Argument " 
								LUACC_NUM_COL("{}")
								//" of " LUACC_FUNCTION "function " 
								//LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING 
								", is " LC_not " a "
								LUACC_SINGLE_STRING("{}")
								,i, slu::getName(arg)
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
			return slu::push(L, std::apply(func, args));
	}


	inline int runCppFuncWrapped(lua_State* L, const std::string& funcName, const auto& func)
	{
		try
		{
			return runCppFunc(L, funcName, func);
		}
		catch (const std::exception& e)
		{
			return slu::lua_error(L, LC_Function " "
				LUACC_SINGLE_STRING("{}")
				" had a " LC_error ": {}"
				, funcName, e.what());
		}
		catch (const slu::Error& e)
		{
			return slu::lua_error(L, LC_Function " "
				LUACC_SINGLE_STRING("{}")
				" had a " LC_error ": {}"
				, funcName, e.msg);
		}
	}



	// Wrap a C++ into a lua function, ment for lua_pushcfunction
#define Slu_wrapRaw(_LUA_NAME,_CPP_FUNC) [](lua_State* L){return ::slu::runCppFuncWrapped(L,_LUA_NAME,_CPP_FUNC);}

	// Wrap a C++ into a lua function, name pair, for your library function tables
#define Slu_wrap(_LUA_NAME,_CPP_FUNC) {_LUA_NAME,Slu_wrapRaw(_LUA_NAME,_CPP_FUNC)}




#define _Slu_MULTI_WRAP_CASE(_LUA_NAME,_COUNT,_CPP_FUNC) case _COUNT:return ::slu::runCppFuncWrapped(L,_LUA_NAME "(" #_COUNT ")",_CPP_FUNC)


	// Wrap 2 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
#define Slu_wrap2Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2) [](lua_State* L){switch(lua_gettop(L)){ \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		default: return ::slu::lua_error(L,LUACC_FUNCTION "Function " \
			LUACC_SINGLE_STRING( _LUA_NAME) \
			" needs " LUACC_NUMBER #_C1 LUACC_DEFAULT " or " LUACC_NUMBER #_C2 " " LC_arguments "." );} }

	// Wrap 2 C++ functions overloaded by arg count into 1 lua function, name pair, for your library function tables
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
#define Slu_wrap2(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2) {_LUA_NAME,Slu_wrap2Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2)}


	// Wrap 3 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
#define Slu_wrap3Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3) [](lua_State* L){switch(lua_gettop(L)){ \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C3,_CPP_FUNC3); \
		default: return ::slu::lua_error(L,LUACC_FUNCTION "Function " \
			LUACC_SINGLE_STRING( _LUA_NAME) \
			" needs " LUACC_NUMBER #_C1 LUACC_DEFAULT \
			", " LUACC_NUMBER #_C2 LUACC_DEFAULT \
			" or " LUACC_NUMBER #_C3 " " LC_arguments "." );} }

	// Wrap 3 C++ functions overloaded by arg count into 1 lua function, name pair, for your library function tables
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
#define Slu_wrap3(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3) {_LUA_NAME,Slu_wrap3Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3)}


	// Wrap 4 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
	// _C4 is the argument count of _CPP_FUNC4
#define Slu_wrap4Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4) [](lua_State* L){switch(lua_gettop(L)){ \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C3,_CPP_FUNC3); \
		_Slu_MULTI_WRAP_CASE(_LUA_NAME,_C4,_CPP_FUNC4); \
		default: return ::slu::lua_error(L,LUACC_FUNCTION "Function " \
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
#define Slu_wrap4(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4) {_LUA_NAME,Slu_wrap4Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4)}
}