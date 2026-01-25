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
#include <cstdint>

#include <slu/ast/OpMacros.hpp>
export module slu.ast.enums;

namespace slu::ast
{
	export enum class OptSafety : uint8_t
	{
		DEFAULT,
		SAFE,
		UNSAFE
	};

	export enum class BinOpType : uint8_t
	{
		NONE,
#define _X(E, T, M) E
		Slu_BIN_OPS(, ),

		//Not a real op, just the amount of binops
		// -1 because NONE, +1 cuz total size
		ENUM_SIZE = AS - 1 + 1
	};

	export enum class UnOpType : uint8_t
	{
		NONE,

		Slu_UN_OPS(, ),

		//Not a real op, just the amount of unops
		//== MUT because NONE isnt a real op.
		ENUM_SIZE = MARK_MUT
	};

	export enum class PostUnOpType : uint8_t
	{
		NONE,

		Slu_POST_UN_OPS(, ),

		//Not a real op, just the amount of post unops
		// -1 because NONE, +1 cuz total size
		ENUM_SIZE = TRY + 1 - 1
	};
} //namespace slu::ast
