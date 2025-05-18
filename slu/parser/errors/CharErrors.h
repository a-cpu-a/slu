/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slua/parser/SimpleErrors.hpp>
#include <slua/parser/Input.hpp>

namespace slua::parse
{
	inline void throwMultilineCommentMissingEqual(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
			"Expected multiline comment, to have atleast "
			LUACC_NUM_COL("1")
			" " LUACC_SINGLE_STRING("=")
			" character between the "
			LUACC_SINGLE_STRING("[")
			" characters"
			+ errorLocStr(in)
		);
	}
	inline void throwUnexpectedVarArgs(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Varargs "
			LUACC_SINGLE_STRING("...")
			" are not supported"
			"{}"
			, errorLocStr(in)
		));
	}
	inline void throwExpectedTraitExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Expected trait expression at"
			"{}"
			, errorLocStr(in)
		));
	}
	inline void throwExpectedStructOrAssign(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Expected struct or assignment at"
			"{}"
			, errorLocStr(in)
		));
	}
	inline void throwExpectedPatDestr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Expected pattern destructuring for "
			LUACC_SINGLE_STRING("as")
			", at"
			"{}"
			, errorLocStr(in)
		));
	}
	inline void throwExpectedTypeExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Expected type expression at"
			"{}"
			, errorLocStr(in)
		));
	}
	inline void throwExpectedExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Expected expression at"
			"{}"
			, errorLocStr(in)
		));
	}
	inline void throwSpaceMissingBeforeString(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Expected space before " LC_string " argument, at"
			"{}"
			, errorLocStr(in)));
	}
	inline void throwSemicolMissingAfterStat(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Expected semicolon (" 
			LUACC_SINGLE_STRING(";")
			") after statment, at"
			"{}"
			, errorLocStr(in)));
	}
	inline void throwVarlistInExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(std::format(
			"Found list of variables inside expression"
			"{}"
			, errorLocStr(in)));
	}
	inline void throwRawExpr(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
			"Raw expressions are " LC_not " allowed, expected assignment or " LC_function " call"
			+ errorLocStr(in));
	}

	//Varlist

	inline void throwFuncCallInVarList(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
			"Cant assign to " LC_function " call (Found in variable list)"
			+ errorLocStr(in));
	}
	inline void throwExprInVarList(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
			"Cant assign to expression (Found in variable list)"
			+ errorLocStr(in));
	}

	//Assign

	inline void throwFuncCallAssignment(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
			"Cant assign to " LC_function " call, found "
			LUACC_SINGLE_STRING("=") " at"
			+ errorLocStr(in));
	}
	inline void throwExprAssignment(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
			"Cant assign to expression, found "
			LUACC_SINGLE_STRING("=") " at"
			+ errorLocStr(in));
	}
	inline void reportIntTooBig(AnyInput auto& in,const std::string_view str)
	{
		in.handleError(std::format(
			LC_Integer " is too big, "
			LUACC_SINGLE_STRING("{}") " at"
			"{}", str, errorLocStr(in)));
	}
	inline void throwUnexpectedFloat(AnyInput auto& in,const std::string_view str)
	{
		in.handleError(std::format(
			"Expected " LC_integer ", found "
			LUACC_SINGLE_STRING("{}") " at"
			"{}", str, errorLocStr(in)));
	}
}