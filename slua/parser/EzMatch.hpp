/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <variant>

//https://thatonegamedev.com/cpp/rust-enums-in-modern-cpp-match-pattern/

// Overloaded utility for std::visit
template<class... Ts>
struct _EzMatchOverloader : Ts... { using Ts::operator()...; };
template<class... Ts>
_EzMatchOverloader(Ts...) -> _EzMatchOverloader<Ts...>;

// Macro Definitions
#define ezmatch(value) [&](auto... __ez_cases) { std::visit(_EzMatchOverloader{__ez_cases...}, value); } // Passes value and cases
#define ezcase(type) [&](type& var)

/*
inline void test()
{
	std::variant<int, double, float> v{ 0.0f };

	ezmatch(v)(

		ezcase(int) { std::cout<<"Int "<<var<<"\n"; },
		ezcase(double) { std::cout<<"Double "<<var<<"\n";},
		ezcase(float) { std::cout<<"Float "<<var<<"\n";}

		);

}
*/