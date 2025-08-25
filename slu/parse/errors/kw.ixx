module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <format>
#include <slu/Ansi.hpp>
export module slu.parse.errors.kw;
import slu.parse.error;
import slu.parse.input;

namespace slu::parse
{
	export void throwExpectedExportable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected exportable statment after " 
			LUACC_SINGLE_STRING("ex")
			", at"
			"{}", errorLocStr(in)));
	}
	export void throwExpectedImplAfterDefer(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected impl statment after " 
			LUACC_SINGLE_STRING("defer")
			", at"
			"{}", errorLocStr(in)));
	}
	export void throwUnexpectedSafety(AnyInput auto& in, const ast::Position pos)
	{
		throw UnexpectedKeywordError(std::format(
			"Unexpected safe/unsafe, at"
			"{}"
			, errorLocStr(in,pos)));
	}
	export void throwExpectedSafeable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected markable statment after " 
			LUACC_SINGLE_STRING("safe")
			", at"
			"{}"
			, errorLocStr(in)));
	}
	export void throwExpectedUnsafeable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected markable statment after " 
			LUACC_SINGLE_STRING("unsafe")
			", at"
			"{}"
			, errorLocStr(in)));
	}
	export void throwExpectedExternable(AnyInput auto& in)
	{
		throw UnexpectedKeywordError(std::format(
			"Expected markable statment after " 
			LUACC_SINGLE_STRING("extern \"...\"")
			", at"
			"{}"
			, errorLocStr(in)));
	}
}