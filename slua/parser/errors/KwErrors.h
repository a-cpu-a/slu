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
		throw UnexpectedCharacterError(std::format(
			"Expected exportable statment after " 
			LUACC_SINGLE_STRING("ex")
			", at"
			"{}"
			, errorLocStr(in)));
	}
}