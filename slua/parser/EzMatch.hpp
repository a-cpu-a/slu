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
#define ezmatch(_VALUE) [&](auto... ez_cases) { std::visit(_EzMatchOverloader{ez_cases...}, _VALUE); } // Passes value and cases
#define ezcase(_TYPE,_VAR) [&](_TYPE& _VAR)
#define ezcaseVal(_TYPE) [&](_TYPE& var)

/*
inline void test()
{
	std::variant<int, double, float> v{ 0.0f };

	ezmatch(v)(

		ezcaseVar(int) { std::cout<<"Int "<< var <<"\n"; },
		ezcaseVar(double) { std::cout<<"Double "<< var <<"\n";},
		ezcase(float,var) { std::cout<<"Float "<< var <<"\n";}

	);
}
*/


/*

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