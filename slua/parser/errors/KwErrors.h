/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slua/parser/SimpleErrors.hpp>
#include <slua/parser/Input.hpp>

namespace sluaParse
{
	inline void throwExpectedExportable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected exportable statment after " 
			LUACC_SINGLE_STRING("ex")
			", at"
			"{}"
			, errorLocStr(in)));
	}
	inline void throwExpectedSafeable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected markable statment after " 
			LUACC_SINGLE_STRING("safe")
			", at"
			"{}"
			, errorLocStr(in)));
	}
	inline void throwExpectedUnsafeable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected markable statment after " 
			LUACC_SINGLE_STRING("unsafe")
			", at"
			"{}"
			, errorLocStr(in)));
	}
}