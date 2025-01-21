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

#include "SkipSpace.hpp"
#include "SimpleErrors.hpp"

namespace sluaParse
{
	template<size_t TOK_SIZE>
	inline [[nodiscard]] void requireToken(AnyInput auto& in, const char(&tok)[TOK_SIZE])
	{
		skipSpace(in);
		try
		{
			for (size_t i = 0; i < TOK_SIZE - 1; i++)//skip null
			{
				if (in.get() != tok[i])
				{
					throw UnexpectedCharacterError(
						"Expected "
						LUACC_START_SINGLE_STRING + std::to_string(tok) + LUACC_END_SINGLE_STRING
						", but found "
						LUACC_START_SINGLE_STRING + tok[i] + LUACC_END_SINGLE_STRING
						+ errorLocStr(in)
					);
				}
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
	template<size_t TOK_SIZE>
	inline [[nodiscard]] bool checkToken(AnyInput auto& in, const char(&tok)[TOK_SIZE], const bool nameLike = false, const bool readIfGood = false)
	{
		size_t off = spacesToSkip(in);

		for (size_t i = 0; i < TOK_SIZE - 1; i++)//skip null
		{
			if (in.checkOOB(off))
				return false;

			if (in.peek(off++) != tok[i])
				return false;
		}


		if (nameLike)
		{
			const uint8_t ch = in.peek(off);
			if (ch == '_' || std::isalnum(ch))
				return false;
		}

		if (readIfGood)
			in.skip(off);


		return true;
	}
	template<size_t TOK_SIZE>
	inline [[nodiscard]] bool checkReadToken(AnyInput auto& in, const char(&tok)[TOK_SIZE], const bool nameLike = false) {
		return checkToken(in, tok, nameLike, true);
	}
	template<size_t TOK_SIZE>
	inline void readOptToken(AnyInput auto& in, const char(&tok)[TOK_SIZE], const bool nameLike = false) {
		(void)checkReadToken(in, tok, nameLike);
	}
	template<size_t TOK_SIZE>
	inline [[nodiscard]] bool checkReadTextToken(AnyInput auto& in, const char(&tok)[TOK_SIZE]) {
		return checkToken(in, tok, true, true);
	}
}