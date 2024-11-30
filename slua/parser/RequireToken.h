/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <luaconf.h>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>

#include "SkipSpace.h"
#include "SimpleErrors.h"

namespace sluaParse
{
	inline void requireToken(AnyInput auto& in, const char* tok)
	{
		try
		{
			size_t i = 0;
			while (tok[i] != 0)
			{
				if (in.get() != tok[i])
					throw UnexpectedCharacterError(
						"Expected "
						LUACC_START_SINGLE_STRING + std::to_string(tok) + LUACC_END_SINGLE_STRING
						", but found "
						LUACC_START_SINGLE_STRING + tok[i] + LUACC_END_SINGLE_STRING
						+ errorLocStr(in)
					);

				i++;
			}
		}
		catch (EndOfStreamError&)
		{
			throw UnexpectedFileEndError(
				"Expected "
				LUACC_START_SINGLE_STRING + std::to_string(tok) + LUACC_END_SINGLE_STRING
				", but file ended");
		}
	}
}