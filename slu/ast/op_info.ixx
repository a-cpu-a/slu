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
module;
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
#define _X(E, T, M) #M##sv
	export constexpr auto unOpNames = mkVar(Slu_UN_OPS(, ));
	export constexpr auto postUnOpNames = mkVar(Slu_POST_UN_OPS(, ));
	export constexpr auto binOpNames = mkVar(Slu_BIN_OPS(, ));
#undef _X
#define _X(E, T, M) #T##sv
	export constexpr auto unOpTraitNames = mkVar(Slu_UN_OPS(, ));
	export constexpr auto postUnOpTraitNames = mkVar(Slu_POST_UN_OPS(, ));
	export constexpr auto binOpTraitNames = mkVar(Slu_BIN_OPS(, ));
} //namespace slu::ast