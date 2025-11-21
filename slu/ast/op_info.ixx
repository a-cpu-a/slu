module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <array>
#include <cstdint>
#include <string_view>

#include <slu/ast/OpMacros.hpp>
export module slu.ast.op_info;

import slu.ast.enums;
import slu.char_info;

namespace slu::ast
{
	using namespace std::string_view_literals;

	template<typename... Ts>
	constexpr auto mkVar(Ts... vars)
	    -> std::array<std::string_view, sizeof...(Ts)>
	{
		return {vars...};
	}
#define STRINGIZE(A) #A
#define _X(E, T, M) STRINGIZE(M)##sv
	export constexpr auto unOpNames = mkVar(Slu_UN_OPS(, ));
	export constexpr auto postUnOpNames = mkVar(Slu_POST_UN_OPS(, ));
	export constexpr auto binOpNames = mkVar(Slu_BIN_OPS(, ));
#undef _X
#define _X(E, T, M) STRINGIZE(T)##sv
	export constexpr auto unOpTraitNames = mkVar(Slu_UN_OPS(, ));
	export constexpr auto postUnOpTraitNames = mkVar(Slu_POST_UN_OPS(, ));
	export constexpr auto binOpTraitNames = mkVar(Slu_BIN_OPS(, ));
} //namespace slu::ast