/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <variant>

//credits to: https://thatonegamedev.com/cpp/rust-enums-in-modern-cpp-match-pattern/

// Overloaded utility for std::visit
template<class... Ts>
struct _EzMatchOverloader : Ts... { using Ts::operator()...; };
template<class... Ts>
_EzMatchOverloader(Ts...) -> _EzMatchOverloader<Ts...>;

// Macro Definitions
#define ezmatch(_VALUE) [&](auto... ez_cases) { return std::visit(_EzMatchOverloader{ez_cases...}, _VALUE); } // Passes value and cases
#define ezcase(_VAR,_TYPE) [&](_TYPE& _VAR)
#define ezcaseVar(_TYPE) [&](_TYPE& var)

/*

#include <iostream>

inline void test()
{
	std::variant<int, double, float> v{ 0.0f };

	ezmatch(v)(// note that this isnt a curly brace!

		ezcaseVar(int) { std::cout<<"Int "<< var <<"\n"; },// note the comma
		ezcaseVar(double) { std::cout<<"Double "<< var <<"\n";},
		ezcase(var,float) { std::cout<<"Float "<< var <<"\n";}// Note the missing comma!

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

		ezcaseVar(MyEnumType::INT) { std::cout<<"Int "<< var <<"\n"; },
		ezcaseVar(double) { std::cout<<"Double "<< var <<"\n";},

		ezcaseVar(MyEnumType::STRNUM) {
			std::cout <<"Strnum "<< var.str 
				<<" N: "<< var.num <<"\n";
		}

	);
}

*/
/*

#include <iostream>

inline int test3()
{
	std::variant<int, bool> v{ 0 };

	// The return type of the cases must be identical!

	// The compiler will try to guess it, so you
	// might need to explicitly state it using "-> Type")

	return ezmatch(v)(
		ezcaseVar(int)-> char { return 1; },
		ezcaseVar(bool)-> char { return 0;}
	);
}

*/