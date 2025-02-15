/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>

namespace slua
{
	//throw inside a check to provide a custom message
	struct Error { std::string msg; };
}


#if !defined(__has_builtin)
#define __has_builtin(X) false
#endif

#if __has_builtin(__builtin_debugtrap)
#define SLua_panic() __builtin_debugtrap()
#elif __has_builtin(__builtin_trap)
#define SLua_panic() __builtin_trap()
#elif defined(_MSC_VER)
#if defined(_M_X64) || defined(_M_I86) || defined(_M_IX86)
#define SLua_panic()__debugbreak() // Smaller
#else
#define SLua_panic() __fastfail(0)
#endif
#else
#define SLua_panic() std::abort()
#endif