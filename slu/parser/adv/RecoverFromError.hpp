/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadStringLiteral.hpp>

namespace slu::parse
{
	inline bool isStrStarter(AnyInput auto& in)
	{
		const char ch1 = in.peek();
		return ch1 == '=' || ch1 == '[' || ch1 == '\'' || ch1 == '"';
	}

	inline bool trySkipMultilineString(AnyInput auto& in)
	{
		const char ch1 = in.peekAt(1);
		if (ch1 == '=' || ch1 == '[')
		{
			readStringLiteral(in, '[');
			return true;
		}
		return false;
	}

	template<size_t TOK_SIZE>
	inline bool recoverErrorTextToken(AnyInput auto& in, const char(&tok)[TOK_SIZE])
	{
		while (in)
		{
			const char ch = in.peek();
			switch (ch)
			{
			case ' ':
			case '\n':
			case '\r':
			case '\t':
			case '\f':
			case '\v':
			case '-':
			{
				skipSpace(in);

				do
				{
					const char chl = in.peek();
					switch (chl)
					{
					case '"':
					case '\'':
						readStringLiteral(in, chl);
						break;
					case '[':
						if (!trySkipMultilineString(in))
							goto break_loop;
						break;
					default:
						goto break_loop;
					}
				} while (skipSpace(in) || isStrStarter(in));
			break_loop:
				break;
			}
			case '"':
			case '\'':
				readStringLiteral(in, ch);
				break;
			case '[':
				trySkipMultilineString(in);
				break;
			default:
				break;
			}

			if (checkTextToken(in, tok))
			{// Found it, recovered!
				return true;
			}
			in.skip();//Not found, try at next char
		}
		return false;
	}
}