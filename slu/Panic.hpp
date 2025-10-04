/*
** See Copyright Notice inside Include.hpp
*/
#pragma once


#if !defined(__has_builtin)
#define __has_builtin(X) false
#endif

#if __has_builtin(__builtin_debugtrap)
#define Slu_panic(...) __builtin_debugtrap()
#elif __has_builtin(__builtin_trap)
#define Slu_panic(...) __builtin_trap()
#elif defined(_MSC_VER)
#if defined(_M_X64) || defined(_M_I86) || defined(_M_IX86)
#define Slu_panic(...) __debugbreak();std::abort() // Smaller
#else
#define Slu_panic(...) __fastfail(0)
#endif
#else
#define Slu_panic(...) std::abort()
#endif

//Runtime checked!
#define Slu_require(COND) do{if(!(COND)){Slu_panic();}}while(false)

//Only in debug builds
#define Slu_assert(COND) Slu_require(COND)
