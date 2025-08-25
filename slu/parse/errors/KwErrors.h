/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parse/SimpleErrors.hpp>
import slu.parse.input;

namespace slu::parse
{
	inline void throwExpectedExportable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected exportable statment after " 
			LUACC_SINGLE_STRING("ex")
			", at"
			"{}", errorLocStr(in)));
	}
	inline void throwExpectedImplAfterDefer(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected impl statment after " 
			LUACC_SINGLE_STRING("defer")
			", at"
			"{}", errorLocStr(in)));
	}
	inline void throwUnexpectedSafety(AnyInput auto& in, const Position pos)
	{
		throw UnexpectedKeywordError(std::format(
			"Unexpected safe/unsafe, at"
			"{}"
			, errorLocStr(in,pos)));
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
	inline void throwExpectedExternable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected markable statment after " 
			LUACC_SINGLE_STRING("extern \"...\"")
			", at"
			"{}"
			, errorLocStr(in)));
	}
}