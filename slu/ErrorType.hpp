/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <format>

namespace slu
{
	//throw inside a check to provide a custom message
	struct Error { 
		std::string msg;

		constexpr Error() = default;
		constexpr Error(const std::string& msg) :msg(msg) {}
		constexpr Error(std::string&& msg) :msg(std::move(msg)) {}

		template<class... Args>
		Error(const std::format_string<Args...> fmt, Args&&... fmtArgs)
			:msg(std::vformat(fmt.get(), std::make_format_args(fmtArgs...)))
		{}
	};
}


#if !defined(__has_builtin)
#define __has_builtin(X) false
#endif

#if __has_builtin(__builtin_debugtrap)
#define Slu_panic(...) __builtin_debugtrap()
#elif __has_builtin(__builtin_trap)
#define Slu_panic(...) __builtin_trap()
#elif defined(_MSC_VER)
#if defined(_M_X64) || defined(_M_I86) || defined(_M_IX86)
#define Slu_panic(...) __debugbreak() // Smaller
#else
#define Slu_panic(...) __fastfail(0)
#endif
#else
#define Slu_panic(...) std::abort()
#endif

//Runtime checked!
#define Slu_require(COND) do{if(!(COND))Slu_panic();}while(false)