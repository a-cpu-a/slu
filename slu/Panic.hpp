/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <print>

#define _Slu_PASS_THROUGH(...) __VA_ARGS__
#define STRINGIZE(...) STRINGIZE2(__VA_ARGS__)
#define STRINGIZE2(...) #__VA_ARGS__

#if !defined(__has_builtin)
#define __has_builtin(X) false
#endif

#define _Slu_panic_msg(...) \
	std::println("Panic in " __FILE__ ":" STRINGIZE(__LINE__) " : " __VA_ARGS__)

#if __has_builtin(__builtin_debugtrap)
#define Slu_panic(...)           \
	_Slu_panic_msg(__VA_ARGS__); \
	__builtin_debugtrap()
#elif __has_builtin(__builtin_trap)
#define Slu_panic(...)           \
	_Slu_panic_msg(__VA_ARGS__); \
	__builtin_trap()
#elif defined(_MSC_VER)
#if defined(_M_X64) || defined(_M_I86) || defined(_M_IX86)
#define Slu_panic(...)            \
	_Slu_panic_msg(__VA_ARGS__);  \
	__debugbreak(); /* Smaller */ \
	std::abort()
#else
#define Slu_panic(...)           \
	_Slu_panic_msg(__VA_ARGS__); \
	__fastfail(0)
#endif
#else
#define Slu_panic(...)           \
	_Slu_panic_msg(__VA_ARGS__); \
	std::abort()
#endif


//Runtime checked!
#define Slu_require(...)    \
	do                      \
	{                       \
		if (!(__VA_ARGS__)) \
		{                   \
			Slu_panic();    \
		}                   \
	} while (false)
#define Slu_requireMsg(MSG, ...) \
	do                           \
	{                            \
		if (!(__VA_ARGS__))      \
		{                        \
			Slu_panic(MSG);      \
		}                        \
	} while (false)

//Only in debug builds
#define Slu_assert(...) Slu_require(__VA_ARGS__)
#define Slu_assertOp(A, OP, B) \
	Slu_requireMsg(_Slu_PASS_THROUGH("Op failed: {} " #OP " {}", A, B), A OP B)
