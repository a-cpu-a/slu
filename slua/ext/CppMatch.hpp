#pragma once
/*
MIT License

Copyright (c) 2025 a-cpu-a

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef _CPP_MATCH_HPP
#define _CPP_MATCH_HPP

#include <variant>

//credits to: https://thatonegamedev.com/cpp/rust-enums-in-modern-cpp-match-pattern/

// Overloaded utility for std::visit
template<class... Ts>
struct _EzMatchOverloader : Ts... { using Ts::operator()...; };
template<class... Ts>
_EzMatchOverloader(Ts...) -> _EzMatchOverloader<Ts...>;

// Macro Definitions
#define ezmatch(_VALUE) [&](auto... ez_cases) \
		{ return ::std::visit(::_EzMatchOverloader{ez_cases...}, _VALUE); }

//Note: you need to add const and &/&& yourself

#define ezcase(_VAR,_TYPE) [&](_TYPE _VAR)
#define varcase(_TYPE) [&](_TYPE var)

#endif //Header guard

/*

#include <iostream>

inline void test()
{
	std::variant<int, double, float> v{ 0.0f };

	ezmatch(v)(// note that this isnt a curly brace!

		varcase(int) { std::cout<<"Int "<< var <<"\n"; },// note the comma
		varcase(double) { std::cout<<"Double "<< var <<"\n";},
		ezcase(varName,float) { std::cout<<"Float "<< varName <<"\n";}
		// Note the missing comma!
	);
}

*/
/*

#include <iostream>

namespace MyEnumType
{
	using INT = int;
	using F64 = double;
	struct STRNUM {std::string str; int num;};
}
using MyEnum = std::variant<
	MyEnumType::INT,
	MyEnumType::F64,
	MyEnumType::STRNUM
>;

inline void test2()
{
	MyEnum v = MyEnumType::STRNUM("aa",0);

	ezmatch(v)(

		varcase(MyEnumType::INT) { std::cout<<"Int "<< var <<"\n"; },
		varcase(double) { std::cout<<"Double "<< var <<"\n";},

		varcase(MyEnumType::STRNUM) {
			std::cout <<"Strnum "<< var.str 
				<<" N: "<< var.num <<"\n";
		}

	);
}

*/
/*

#include <iostream>

consteval int test3()
{
	std::variant<int, bool> v{ 0 };

	// The return types of the cases must be identical.

	// The compiler will add the return type itself, so
	// using "-> Type" is optional, but required in some cases

	return ezmatch(v)(
		varcase(int) -> char { return 1; },
		varcase(bool) -> char { return 0;}
	);
}

*/
