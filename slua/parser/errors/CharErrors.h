/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slua/parser/SimpleErrors.hpp>
#include <slua/parser/Input.hpp>

namespace sluaParse
{
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
			LUACC_SINGLE_STRING("=")
			+ errorLocStr(in));
	}
	inline void throwExprAssignment(AnyInput auto& in)
	{
		throw UnexpectedCharacterError(
			"Cant assign to expression, found "
			LUACC_SINGLE_STRING("=")
			+ errorLocStr(in));
	}
	inline void throwIntTooBig(AnyInput auto& in,const std::string_view str)
	{
		throw UnexpectedCharacterError(std::format(
			LC_Integer " is too big, "
			LUACC_SINGLE_STRING("{}")
			"{}", str, errorLocStr(in)));
	}
}