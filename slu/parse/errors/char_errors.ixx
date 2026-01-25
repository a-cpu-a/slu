/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
#include <format>

#include <slu/Ansi.hpp>
export module slu.parse.errors.char_errors;
import slu.parse.error;
import slu.parse.input;

namespace slu::parse
{
	export void throwUnexpectedVarArgs(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
		    "Varargs " LUACC_SINGLE_STRING("...") " are not supported"
		                                          "{}",
		    errorLocStr(in)));
	}
	export void throwExpectedTraitExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    std::format("Expected trait expression at"
		                "{}",
		        errorLocStr(in)));
	}
	export void throwExpectedStructOrAssign(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    std::format("Expected struct or assignment at"
		                "{}",
		        errorLocStr(in)));
	}
	export void throwExpectedPatDestr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
		    "Expected pattern destructuring for " LUACC_SINGLE_STRING(
		        "as") ", at"
		              "{}",
		    errorLocStr(in)));
	}
	export void throwExpectedTypeExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format("Expected type expression at"
		                                           "{}",
		    errorLocStr(in)));
	}
	export void throwExpectedExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format("Expected expression at"
		                                           "{}",
		    errorLocStr(in)));
	}
	export void throwSpaceMissingBeforeString(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    std::format("Expected space before " LC_string " argument, at"
		                "{}",
		        errorLocStr(in)));
	}
	export void throwSemicolMissingAfterStat(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    std::format("Expected semicolon (" LUACC_SINGLE_STRING(
		                    ";") ") after statment, at"
		                         "{}",
		        errorLocStr(in)));
	}
	export void throwVarlistInExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    std::format("Found list of variables inside expression"
		                "{}",
		        errorLocStr(in)));
	}
	export void throwRawExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    "Raw expressions are " LC_not
		    " allowed, expected assignment or " LC_function " call"
		    + errorLocStr(in));
	}

	//Varlist

	export void throwFuncCallInVarList(AnyInput auto& in)
	{
		throw UnexpectedCharacterError("Cant assign to " LC_function
		                               " call (Found in variable list)"
		    + errorLocStr(in));
	}
	export void throwExprInVarList(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    "Cant assign to expression (Found in variable list)"
		    + errorLocStr(in));
	}

	//Assign

	export void throwFuncCallAssignment(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    "Cant assign to " LC_function
		    " call, found " LUACC_SINGLE_STRING("=") " at"
		    + errorLocStr(in));
	}
	export void throwExprAssignment(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
		    "Cant assign to expression, found " LUACC_SINGLE_STRING("=") " at"
		    + errorLocStr(in));
	}
	export void reportIntTooBig(AnyInput auto& in, const std::string_view str)
	{
		in.handleError(std::format(LC_Integer
		    " is too big, " LUACC_SINGLE_STRING("{}") " at"
		                                              "{}",
		    str, errorLocStr(in)));
	}
	export void throwUnexpectedFloat(
	    AnyInput auto& in, const std::string_view str)
	{
		in.handleError(std::format("Expected " LC_integer
		                           ", found " LUACC_SINGLE_STRING("{}") " at"
		                                                                "{}",
		    str, errorLocStr(in)));
	}
} //namespace slu::parse
