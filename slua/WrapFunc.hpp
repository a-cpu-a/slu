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

#include <slua/types/complex/ByteArray.hpp>
#include <slua/types/complex/Function.hpp>
#include <slua/types/complex/IntRef.hpp>
#include <slua/types/complex/OptFunc.hpp>
#include <slua/types/complex/String.hpp>
#include <slua/types/complex/TableKey.hpp>*/

namespace slua
{

	template<typename RET_T>
	using Args0CppFunc = RET_T(*)();
	template<typename RET_T, typename IN1_T>
	using Args1CppFunc = RET_T(*)(IN1_T&);

	template<typename RET_T, typename IN1_T, typename IN2_T>
	using Args2CppFunc = RET_T(*)(IN1_T&, IN2_T&);
	template<typename RET_T, typename IN1_T, typename IN2_T, typename IN3_T>
	using Args3CppFunc = RET_T(*)(IN1_T&, IN2_T&, IN3_T&);
	template<typename RET_T, typename IN1_T, typename IN2_T, typename IN3_T, typename IN4_T>
	using Args4CppFunc = RET_T(*)(IN1_T&, IN2_T&, IN3_T&, IN4_T&);
	template<typename RET_T, typename IN1_T, typename IN2_T, typename IN3_T, typename IN4_T, typename IN5_T>
	using Args5CppFunc = RET_T(*)(IN1_T&, IN2_T&, IN3_T&, IN4_T&, IN5_T&);
	template<typename RET_T, typename IN1_T, typename IN2_T, typename IN3_T, typename IN4_T, typename IN5_T, typename IN6_T>
	using Args6CppFunc = RET_T(*)(IN1_T&, IN2_T&, IN3_T&, IN4_T&, IN5_T&, IN6_T&);


#define _SLua_REQUIRE_ARGS(_COUNT) \
	if(lua_gettop(L)!=_COUNT) \
		return slua::lua_error(L, LUACC_FUNCTION "Function " \
			LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING \
			" needs " LUACC_NUMBER #_COUNT LUACC_ARGUMENT " arguments" LUACC_DEFAULT ".")

#define _SLua_CHECK_ARG(_ARG) \
	const bool arg_ ## _ARG ## _res = slua::checkThrowing<IN ## _ARG ## _T>(L, _ARG); \
	if (!arg_ ## _ARG ## _res) \
		throw slua::Error(LUACC_ARGUMENT "Argument " \
			LUACC_NUMBER #_ARG LUACC_DEFAULT " of " LUACC_FUNCTION "function " \
			LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING \
			", is " LUACC_INVALID "not" LUACC_DEFAULT " a " \
			LUACC_END_SINGLE_STRING + slua::getName<IN ## _ARG ## _T>() + LUACC_STRING_SINGLE "'" \
		); /*\
	if(arg_ ## _ARG ## _res>0) return arg_ ## _ARG ## _res*/ //why did this exist, lol

#define _SLua_READ_ARG(_ARG) IN ## _ARG ## _T in ## _ARG = slua::read<IN ## _ARG ## _T>(L, _ARG)
#define _SLua_CHECK_N_READ_ARG(_ARG) _SLua_CHECK_ARG(_ARG); _SLua_READ_ARG(_ARG)

#define _SLua_RUN_CPP_FUNC(_VAR,...) const slua::ToLua<RET_T> _VAR = func(__VA_ARGS__)
#define _SLua_RETURN_VALUE(_VAR)  return slua::ToLua<RET_T>::push(L,_VAR)

#define _SLua_RUN_N_RETURN(...) _SLua_RUN_CPP_FUNC(ret,__VA_ARGS__); _SLua_RETURN_VALUE(ret)

	template <typename RET_T>
	inline int runCppFunc(lua_State* L, const std::string& funcName, Args0CppFunc<RET_T> func)
	{
		_SLua_REQUIRE_ARGS(0);
		_SLua_RUN_N_RETURN();
	}
	template <typename RET_T, typename IN1_T>
	inline int runCppFunc(lua_State* L, const std::string& funcName, Args1CppFunc< RET_T, IN1_T> func)
		requires (!std::is_same_v<IN1_T, slua::Context>)
	{
		if (lua_gettop(L) != 1)
			return slua::lua_error(L, LUACC_FUNCTION "Function "
				LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
				" needs " LUACC_NUMBER "1" LUACC_ARGUMENT " argument" LUACC_DEFAULT "."
			);

		_SLua_CHECK_N_READ_ARG(1);

		_SLua_RUN_N_RETURN(in1);
	}

	template <typename RET_T, typename IN1_T, typename IN2_T>
	inline int runCppFunc(lua_State* L, const std::string& funcName, Args2CppFunc<RET_T, IN1_T, IN2_T> func)
	{
		_SLua_REQUIRE_ARGS(2);

		_SLua_CHECK_N_READ_ARG(1);
		_SLua_CHECK_N_READ_ARG(2);

		_SLua_RUN_N_RETURN(in1, in2);
	}
	template <typename RET_T, typename IN1_T, typename IN2_T, typename IN3_T>
	inline int runCppFunc(lua_State* L, const std::string& funcName, Args3CppFunc<RET_T, IN1_T, IN2_T, IN3_T> func)
	{
		_SLua_REQUIRE_ARGS(3);

		_SLua_CHECK_N_READ_ARG(1);
		_SLua_CHECK_N_READ_ARG(2);
		_SLua_CHECK_N_READ_ARG(3);

		_SLua_RUN_N_RETURN(in1, in2, in3);
	}
	template <typename RET_T, typename IN1_T, typename IN2_T, typename IN3_T, typename IN4_T>
	inline int runCppFunc(lua_State* L, const std::string& funcName, Args4CppFunc<RET_T, IN1_T, IN2_T, IN3_T, IN4_T> func)
	{
		_SLua_REQUIRE_ARGS(4);

		_SLua_CHECK_N_READ_ARG(1);
		_SLua_CHECK_N_READ_ARG(2);
		_SLua_CHECK_N_READ_ARG(3);
		_SLua_CHECK_N_READ_ARG(4);

		_SLua_RUN_N_RETURN(in1, in2, in3, in4);
	}
	template <typename RET_T, typename IN1_T, typename IN2_T, typename IN3_T, typename IN4_T, typename IN5_T>
	inline int runCppFunc(lua_State* L, const std::string& funcName, Args5CppFunc<RET_T, IN1_T, IN2_T, IN3_T, IN4_T, IN5_T> func)
	{
		_SLua_REQUIRE_ARGS(5);

		_SLua_CHECK_N_READ_ARG(1);
		_SLua_CHECK_N_READ_ARG(2);
		_SLua_CHECK_N_READ_ARG(3);
		_SLua_CHECK_N_READ_ARG(4);
		_SLua_CHECK_N_READ_ARG(5);

		_SLua_RUN_N_RETURN(in1, in2, in3, in4, in5);
	}
	template <typename RET_T, typename IN1_T, typename IN2_T, typename IN3_T, typename IN4_T, typename IN5_T, typename IN6_T>
	inline int runCppFunc(lua_State* L, const std::string& funcName, Args6CppFunc<RET_T, IN1_T, IN2_T, IN3_T, IN4_T, IN5_T, IN6_T> func)
	{
		_SLua_REQUIRE_ARGS(6);

		_SLua_CHECK_N_READ_ARG(1);
		_SLua_CHECK_N_READ_ARG(2);
		_SLua_CHECK_N_READ_ARG(3);
		_SLua_CHECK_N_READ_ARG(4);
		_SLua_CHECK_N_READ_ARG(5);
		_SLua_CHECK_N_READ_ARG(6);

		_SLua_RUN_N_RETURN(in1, in2, in3, in4, in5, in6);
	}

	template<class RET_T, class... ARGS>
	using CtxCppFunc = RET_T(*)(slua::Context& ctx, ARGS...);

	//Throws std::exception, slua::Error
	template <class RET_T,class... ARGS>
	inline int runCppFunc(lua_State* L, const std::string& funcName, CtxCppFunc<RET_T,ARGS...> func)
	{
		if (sizeof...(ARGS) != lua_gettop(L))
		{
			if constexpr (sizeof...(ARGS) == 1)
			{
				return slua::lua_error(L, LUACC_FUNCTION "Function "
					LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
					" needs " LUACC_NUMBER "1" LUACC_ARGUMENT " argument" LUACC_DEFAULT ".");
			}
			return slua::lua_error(L, LUACC_FUNCTION "Function "
				LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
				" needs " LUACC_NUMBER + TS(sizeof...(ARGS)) + LUACC_ARGUMENT " arguments" LUACC_DEFAULT ".");
		}



		//TODO: replace the following with a template/constexpr for loop... or just reflection maybe
		std::tuple<slua::Context, ARGS...> args;

		int i = 0;
		std::apply(
			[&](auto&... argss) {(
				[&](auto& arg) {
					if constexpr (std::is_same_v<decltype(arg), slua::Context&>)//NOTE: reference to pointer
						arg= L;
					else
					{
						i++;
						if (slua::checkThrowing(L, i, arg))
							arg = slua::read(L, i, arg);
						else
							throw slua::Error(
								LUACC_ARGUMENT "Argument " 
								LUACC_NUMBER +TS(i) + LUACC_DEFAULT 
								//" of " LUACC_FUNCTION "function " 
								//LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING 
								", is " LUACC_INVALID "not" LUACC_DEFAULT " a " 
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

	inline int runCppFuncWrapped(lua_State* L, const std::string& funcName, auto func)
	{
		try
		{
			return runCppFunc(L, funcName, func);
		}
		catch (const std::exception& e)
		{
			return slua::lua_error(L, LUACC_FUNCTION "Function "
				LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
				" had a " LUACC_INVALID "error" LUACC_DEFAULT ": "
				, e.what());
		}
		catch (const slua::Error& e)
		{
			return slua::lua_error(L, LUACC_FUNCTION "Function "
				LUACC_START_SINGLE_STRING + funcName + LUACC_END_SINGLE_STRING
				" had a " LUACC_INVALID "error" LUACC_DEFAULT ": "
				, e);
		}
	}



	// Wrap a C++ into a lua function, ment for lua_pushcfunction
#define SLua_WrapRaw(_LUA_NAME,_CPP_FUNC) [](lua_State* L){return ::slua::runCppFuncWrapped(L,_LUA_NAME,_CPP_FUNC);}

	// Wrap a C++ into a lua function, name pair, for your library function tables
#define SLua_Wrap(_LUA_NAME,_CPP_FUNC) {_LUA_NAME,SLua_WrapRaw(_LUA_NAME,_CPP_FUNC)}




#define _SLua_MULTI_WRAP_CASE(_LUA_NAME,_COUNT,_CPP_FUNC) case _COUNT:return ::slua::runCppFuncWrapped(L,_LUA_NAME "(" #_COUNT ")",_CPP_FUNC)


	// Wrap 2 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
#define SLua_Wrap2Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2) [](lua_State* L){switch(lua_gettop(L)){ \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		default: return ::slua::lua_error(L,LUACC_FUNCTION "Function " \
			LUACC_START_SINGLE_STRING _LUA_NAME LUACC_END_SINGLE_STRING \
			" needs " LUACC_NUMBER #_C1 LUACC_DEFAULT " or " LUACC_NUMBER #_C2 " " LUACC_ARGUMENT "arguments" LUACC_DEFAULT "." );} }

	// Wrap 2 C++ functions overloaded by arg count into 1 lua function, name pair, for your library function tables
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
#define SLua_Wrap2(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2) {_LUA_NAME,SLua_Wrap2Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2)}


	// Wrap 3 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
#define SLua_Wrap3Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3) [](lua_State* L){switch(lua_gettop(L)){ \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C3,_CPP_FUNC3); \
		default: return ::slua::lua_error(L,LUACC_FUNCTION "Function " \
			LUACC_START_SINGLE_STRING _LUA_NAME LUACC_END_SINGLE_STRING \
			" needs " LUACC_NUMBER #_C1 LUACC_DEFAULT \
			", " LUACC_NUMBER #_C2 LUACC_DEFAULT \
			" or " LUACC_NUMBER #_C3 " " LUACC_ARGUMENT "arguments" LUACC_DEFAULT "." );} }

	// Wrap 3 C++ functions overloaded by arg count into 1 lua function, name pair, for your library function tables
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
#define SLua_Wrap3(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3) {_LUA_NAME,SLua_Wrap3Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3)}


	// Wrap 4 C++ functions overloaded by arg count into 1 lua function, ment for lua_pushcfunction
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
	// _C4 is the argument count of _CPP_FUNC4
#define SLua_Wrap4Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4) [](lua_State* L){switch(lua_gettop(L)){ \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C1,_CPP_FUNC); \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C2,_CPP_FUNC2); \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C3,_CPP_FUNC3); \
		_SLua_MULTI_WRAP_CASE(_LUA_NAME,_C4,_CPP_FUNC4); \
		default: return ::slua::lua_error(L,LUACC_FUNCTION "Function " \
			LUACC_START_SINGLE_STRING _LUA_NAME LUACC_END_SINGLE_STRING \
			" needs " LUACC_NUMBER #_C1 LUACC_DEFAULT \
			", " LUACC_NUMBER #_C2 LUACC_DEFAULT \
			", " LUACC_NUMBER #_C3 LUACC_DEFAULT \
			" or " LUACC_NUMBER #_C4 " " LUACC_ARGUMENT "arguments" LUACC_DEFAULT "." );} }

	// Wrap 4 C++ functions overloaded by arg count into 1 lua function, name pair, for your library function tables
	// _C1 is the argument count of _CPP_FUNC
	// _C2 is the argument count of _CPP_FUNC2
	// _C3 is the argument count of _CPP_FUNC3
	// _C4 is the argument count of _CPP_FUNC4
#define SLua_Wrap4(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4) {_LUA_NAME,SLua_Wrap4Raw(_LUA_NAME,_C1,_CPP_FUNC,_C2,_CPP_FUNC2,_C3,_CPP_FUNC3,_C4,_CPP_FUNC4)}
}