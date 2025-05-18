/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/Input.hpp>
#include <slua/parser/ManageNewline.hpp>
#include <slua/paint/SemOutputStream.hpp>

namespace slua::parse
{
	constexpr bool isSpaceChar(const char ch)
	{
		return ch == ' ' || ch == '\f' || ch == '\t' || ch == '\v' || ch == '\n' || ch == '\r';
	}

	//Returns idx of non-space/comment char
	inline size_t weakSkipSpace(AnyInput auto& in, size_t idx)
	{
		while (true)
		{
			//start by skipping ez stuff
			while (isSpaceChar(in.peekAt(idx))) { idx++; }
			//then hard stuff...

			if (in.peekAt(idx) != '-' || in.peekAt(idx + 1) != '-')
				break;
			idx += 2;//skip --

			char ch = in.peekAt(idx);
			if (ch == '[')
			{
				size_t level = 0;
				while (in.peekAt(idx + 1 + level) == '=') level++;


				if constexpr (in.settings() & sluaSyn)
				{
					if (level == 0)
						throwMultilineCommentMissingEqual(in);
				}


				if (in.peekAt(idx + 1 + level) == '[') // Confirm multiline comment
				{
					idx += 2 + level; // Skip opening '[=['

					// Consume until closing long bracket
					while (true)
					{
						if (in.peekAt(idx) == ']')
						{
							size_t closeLevel = 0;
							while (in.peekAt(idx + 1 + closeLevel) == '=') closeLevel++;

							if (in.peekAt(idx + 1 + closeLevel) == ']' && closeLevel == level)
							{
								idx += 2 + closeLevel; // Skip closing ']=]'
								break;
							}
						}
						idx++;
					}
					break;//no longer single line comment
				}
			}
			// Otherwise, it's a single-line comment
			while (true)
			{
				if (ch == '\n' || ch == '\r')
					break;
				idx++; // Skip until newline or multiline comment
				ch = in.peekAt(idx);//Get next char
			}
			
			//try skipping normal space again
		}
		return idx;
	}

	template <typename T>
	concept InputOrSemTokStream = AnyInput<T> || slua::paint::AnySemOutput<T>;

	//Returns true, when idx changed
	template<InputOrSemTokStream SorIn>
	inline bool skipSpace(SorIn& sOrIn)
	{
		constexpr bool isSem = slua::paint::AnySemOutput<SorIn>;
		auto& in = *([&] {
			if constexpr (isSem)
				return &sOrIn.in;
			else
				return &sOrIn;
			})();
		
		/*

		Lua is a free-form language.
		It ignores spaces and comments between lexical
		elements (tokens), except as delimiters between
		two tokens.


		Multiline comments:

		We define an opening long bracket of level n as
		an opening square bracket followed by n equal
		signs followed by another opening square bracket.

		So, an opening long bracket of level 0 is written
		as [[, an opening long bracket of level 1 is
		written as [=[, and so on.

		A closing long bracket is defined similarly; for
		instance, a closing long bracket of level 4 is
		written as ]====].

		A long literal starts with
		an opening long bracket of any level and ends at
		the first closing long bracket of the same level.

		It can contain any text except a closing bracket
		of the same level.

		Literals in this bracketed form can run for
		several lines, do not interpret any escape
		sequences, and ignore long brackets of
		any other level.

		*/

		bool res = false;

		bool insideLineComment = false;
		ParseNewlineState nlState = ParseNewlineState::NONE;

		bool insideMultilineComment = false;
		size_t multilineCommentLevel = SIZE_MAX;

		while (in)
		{
			const uint8_t ch = in.peek();
			const bool newLine = manageNewlineState<true>(ch, nlState, in);

			if (insideLineComment)
			{
				if (newLine)
				{
					insideLineComment = false;//new line, so exit comment
					continue;
				}

				if constexpr (isSem)
					sOrIn.template add<paint::Tok::WHITESPACE>();
				in.skip();
				continue;
			}

			// Handle inside-multiline-comment scenario
			else if (insideMultilineComment)
			{
				if (newLine)
					continue;

				if (ch == ']') // Check for possible multiline comment closing
				{
					size_t level = 0;
					while (in.peekAt(1 + level) == '=') // Count '=' signs
						level++;


					if (in.peekAt(1 + level) == ']' && level == multilineCommentLevel)
					{
						insideMultilineComment = false;
						if constexpr (isSem)
							sOrIn.template add<paint::Tok::COMMENT_OUTER>(2 + level);
						in.skip(2 + level); // Skip closing bracket
						continue;
					}
				}
				if constexpr (isSem)
					sOrIn.template add<paint::Tok::WHITESPACE>();
				in.skip(); // Skip other characters in multiline comment
				continue;
			}

			/*

			In source code, Lua recognizes as
			spaces the standard ASCII whitespace characters
			space, form feed, newline, carriage return,
			horizontal tab, and vertical tab.

			*/

			if (newLine)
			{
				res = true;
				continue;
			}

			if (isSpaceChar(ch))
			{
				if constexpr (isSem)
					sOrIn.template add<paint::Tok::WHITESPACE>();
				in.skip();
				res = true;
				continue;
			}

			if (ch == '-')
			{//maybe comment?
				const uint8_t nextCh = in.peekAt(1);
				if (nextCh == '-') // Single-line or multiline comment starts
				{
					res = true;

					const uint8_t nextNextCh = in.peekAt(2);
					if (nextNextCh == '[') // Possible multiline comment
					{
						size_t level = 0;
						while (in.peekAt(3 + level) == '=') // Count '=' signs
							level++;

						if constexpr (SorIn::settings() & sluaSyn && !isSem)
						{
							if (level == 0)
								throwMultilineCommentMissingEqual(in);
						}

						if (in.peekAt(3 + level) == '[') // Confirm multiline comment start
						{
							insideMultilineComment = true;
							multilineCommentLevel = level;
							if constexpr (isSem)
								sOrIn.template add<paint::Tok::COMMENT_OUTER>(4 + level);
							in.skip(4 + level); // Skip "--[=["
							continue;
						}
					}

					insideLineComment = true; // Otherwise, it's a single-line comment
					
					if constexpr (isSem)
						sOrIn.template add<paint::Tok::COMMENT_OUTER>(2);

					in.skip(2); // Skip "--"
					continue;
				}
			}
			break;
		}

		return res;
	}
}