/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
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
	__builtin_debugtrap();       \
	std::abort()
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
