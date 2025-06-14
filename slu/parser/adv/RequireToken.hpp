﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <format>
#include <cstdint>
#include <slu/ext/lua/luaconf.h>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/SimpleErrors.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/basic/CharInfo.hpp>

namespace slu::parse
{
	template<bool SKIP_SPACE=true,size_t TOK_SIZE>
	inline void requireToken(AnyInput auto& in, const char(&tok)[TOK_SIZE])
	{
		if constexpr(SKIP_SPACE)
			skipSpace(in);

		try
		{
			for (size_t i = 0; i < TOK_SIZE - 1; i++)//skip null
			{
				const char ch = in.get();
				if (ch != tok[i])
				{
					throw UnexpectedCharacterError(std::format(
						"Expected "
						LUACC_SINGLE_STRING("{}")
						", character "
						LUACC_SINGLE_STRING("{}")
						", but found "
						LUACC_SINGLE_STRING("{}")
						"{}"
					, tok, tok[i], ch, errorLocStr(in)));
				}
			}
		}
		catch (EndOfStreamError&)
		{
			throw UnexpectedFileEndError(std::format(
				"Expected "
				LUACC_SINGLE_STRING("{}")
				", but file ended",tok));
		}
	}
	//Will NOT skip space!!!
	template<size_t TOK_SIZE>
	[[nodiscard]] inline bool checkToken(AnyInput auto& in, const char(&tok)[TOK_SIZE], const bool nameLike = false, const bool readIfGood = false)
	{
		size_t off = 0;

		for (size_t i = 0; i < TOK_SIZE - 1; i++)//skip null
		{
			if (in.isOob(off))
				return false;

			if (in.peekAt(off++) != tok[i])
				return false;
		}


		if (nameLike)
		{
			if (!in.isOob(off))
			{

				const uint8_t ch = in.peekAt(off);
				if (isValidNameChar(ch))
					return false;
			}
		}

		if (readIfGood)
			in.skip(off);


		return true;
	}
	//Will NOT skip space!!!
	template<size_t TOK_SIZE>
	[[nodiscard]] inline bool checkTextToken(AnyInput auto& in, const char(&tok)[TOK_SIZE]) {
		return checkToken(in, tok, true);
	}

	template<size_t TOK_SIZE>
	[[nodiscard]] inline bool checkReadToken(AnyInput auto& in, const char(&tok)[TOK_SIZE], const bool nameLike = false) {
		skipSpace(in);
		return checkToken(in, tok, nameLike, true);
	}
	template<size_t TOK_SIZE>
	[[nodiscard]] inline bool checkReadTextToken(AnyInput auto& in, const char(&tok)[TOK_SIZE]) {
		skipSpace(in);
		return checkToken(in, tok, true, true);
	}

	template<size_t TOK_SIZE>
	inline void readOptToken(AnyInput auto& in, const char(&tok)[TOK_SIZE], const bool nameLike = false) {
		(void)checkReadToken(in, tok, nameLike);
	}
}