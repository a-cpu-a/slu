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
#include <format>

#include <slu/Ansi.hpp>
export module slu.parse.errors.kw;
import slu.ast.pos;
import slu.parse.error;
import slu.parse.input;

namespace slu::parse
{
	export void throwExpectedExportable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
		    "Expected exportable statment after " LUACC_SINGLE_STRING(
		        "ex") ", at"
		              "{}",
		    errorLocStr(in)));
	}
	export void throwExpectedImplAfterDefer(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
		    "Expected impl statment after " LUACC_SINGLE_STRING("defer") ", at"
		                                                                 "{}",
		    errorLocStr(in)));
	}
	export void throwUnexpectedSafety(
	    AnyInput auto& in, const ast::Position pos)
	{
		throw UnexpectedKeywordError(std::format("Unexpected safe/unsafe, at"
		                                         "{}",
		    errorLocStr(in, pos)));
	}
	export void throwExpectedSafeable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(
		    std::format("Expected markable statment after " LUACC_SINGLE_STRING(
		                    "safe") ", at"
		                            "{}",
		        errorLocStr(in)));
	}
	export void throwExpectedUnsafeable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(
		    std::format("Expected markable statment after " LUACC_SINGLE_STRING(
		                    "unsafe") ", at"
		                              "{}",
		        errorLocStr(in)));
	}
	export void throwExpectedExternable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(
		    std::format("Expected markable statment after " LUACC_SINGLE_STRING(
		                    "extern \"...\"") ", at"
		                                      "{}",
		        errorLocStr(in)));
	}
} //namespace slu::parse