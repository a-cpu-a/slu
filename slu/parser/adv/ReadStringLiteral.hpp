/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <format>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>

namespace slu::parse
{
	/**
	 * Encode a code point using UTF-8
	 *
	 * @author Ondřej Hruška <ondra@ondrovo.com> 
	 * @author 31 bit support added by a-cpu-a | credits: https://github.com/ww898/utf-cpp/blob/master/include/ww898/cp_utf8.hpp
	 * @license MIT
	 *
	 * @param utf - code point 0-0x7FFFFFFF
	 * @returns U+FFFD on failure (3 bytes)
	 */
	inline std::string u32ToUtf8(const uint32_t utf)
	{
		if (utf <= 0x7F)
			// Plain ASCII
			return std::string(1, (char)utf);
		else if (utf <= 0x07FF)
		{// 2-byte unicode
			char out[2];

			out[0] = char(((utf >> 6) & 0x1F) | 0xC0);
			out[1] = char(((utf) & 0x3F) | 0x80);

			return std::string(&out[0], &out[2]);
		}
		else if (utf <= 0xFFFF)
		{// 3-byte unicode
			char out[3];

			out[0] = char(((utf >> 12) & 0x0F) | 0xE0);
			out[1] = char(((utf >> 6) & 0x3F) | 0x80);
			out[2] = char(((utf) & 0x3F) | 0x80);

			return std::string(&out[0], &out[3]);
		}
		else if (utf <= 0x1FFFFF)
		{// 4-byte unicode
			char out[4];

			out[0] = char(((utf >> 18) & 0x07) | 0xF0);
			out[1] = char(((utf >> 12) & 0x3F) | 0x80);
			out[2] = char(((utf >> 6) & 0x3F) | 0x80);
			out[3] = char(((utf) & 0x3F) | 0x80);

			return std::string(&out[0], &out[4]);
		}
		else if (utf <= 0x3FFFFFF)
		{// 5-byte unicode
			char out[5];

			out[0] = char(((utf >> 24) & 0xFF) | 0xF8);
			out[1] = char(((utf >> 18) & 0x3F) | 0x80);
			out[2] = char(((utf >> 12) & 0x3F) | 0x80);
			out[3] = char(((utf >> 6) & 0x3F) | 0x80);
			out[4] = char(((utf) & 0x3F) | 0x80);

			return std::string(&out[0], &out[5]);
		}
		else if (utf <= 0x7FFFFFFF)
		{// 6-byte unicode
			char out[6];

			out[0] = char(((utf >> 30) & 0xFF) | 0xFC);
			out[1] = char(((utf >> 24) & 0x3F) | 0x80);
			out[2] = char(((utf >> 18) & 0x3F) | 0x80);
			out[3] = char(((utf >> 12) & 0x3F) | 0x80);
			out[4] = char(((utf >> 6) & 0x3F) | 0x80);
			out[5] = char(((utf) & 0x3F) | 0x80);

			return std::string(&out[0], &out[6]);
		}
		else
		{// error - use replacement character
			_ASSERT(false);//!!
			const uint8_t out[3] = { 0xEF,0xBF,0xBD };

			return std::string(&out[0], &out[3]);
		}
	}
	/*
		// NOTE: more ='s can be added, But both sides must have the same amount (closing things inside, with different counts are treated as normal text)
		// NOTE: NO WHITESPACE BETWEEN CHARS!!!
		// NOTE: approximate!!
		
		StringLiteral ::= "\"" StringLiteralInner "\"" | "'" StringLiteralInner "'" | "[[" {NormalCharacter} "]]" | "[=[" {NormalCharacter} "]=]" | "[==[" {NormalCharacter} "]==]"

		StringLiteralInner ::= {Char}

		Char ::= NormalCharacter | "\\" | "\a" | "\b" | "\f" | "\n" | "\r" | "\t" | "\v" | "\"" | "\'" | "\z" | HexEscape | U8Escape | DecEscape

		HexEscape ::= "\x" HexDigit HexDigit
		U8Escape ::= "\u{" [[[[[[[Digits_0_To_8] HexDigit] HexDigit] HexDigit] HexDigit] HexDigit] HexDigit] HexDigit "}"
		DecEscape ::= "\" Digit [Digit [Digit]]


	*/
	/*
		A short literal string can be delimited by matching single or
		double quotes, and can contain the following C-like escape 
		sequences: '\a' (bell), '\b' (backspace), '\f' (form feed),
		'\n' (newline), '\r' (carriage return), '\t' (horizontal tab),
		'\v' (vertical tab), '\\' (backslash),
		'\"' (quotation mark [double quote]), and 
		'\'' (apostrophe [single quote]). A backslash followed by a 
		line break results in a newline in the string. The escape 
		sequence '\z' skips the following span of whitespace 
		characters, including line breaks; it is particularly 
		useful to break and indent a long literal string into multiple 
		lines without adding the newlines and spaces into the string 
		contents. A short literal string cannot contain unescaped line 
		breaks nor escapes not forming a valid escape sequence.

		We can specify any byte in a short literal string, including 
		embedded zeros, by its numeric value. This can be done with 
		the escape sequence \xXX, where XX is a sequence of exactly 
		two hexadecimal digits, or with the escape sequence \ddd, 
		where ddd is a sequence of up to three decimal digits. (Note 
		that if a decimal escape sequence is to be followed by a 
		digit, it must be expressed using exactly three digits.)
		
		The UTF-8 encoding of a Unicode character can be inserted 
		in a literal string with the escape sequence \u{XXX} (with 
		mandatory enclosing braces), where XXX is a sequence of one 
		or more hexadecimal digits representing the character code 
		point. This code point can be any value less than 2^31. (Lua 
		uses the original UTF-8 specification here, which is not 
		restricted to valid Unicode code points.)
		
		Literal strings can also be defined using a long format 
		enclosed by long brackets. We define an opening long bracket 
		of level n as an opening square bracket followed by n equal 
		signs followed by another opening square bracket. So, an 
		opening long bracket of level 0 is written as [[, an opening 
		long bracket of level 1 is written as [=[, and so on. A 
		closing long bracket is defined similarly; for instance, a 
		closing long bracket of level 4 is written as ]====]. A 
		long literal starts with an opening long bracket of any level 
		and ends at the first closing long bracket of the same level. 
		It can contain any text except a closing bracket of the same level. 
		Literals in this bracketed form can run for several lines, do not 
		interpret any escape sequences, and ignore long brackets of any 
		other level. Any kind of end-of-line sequence (carriage return, 
		newline, carriage return followed by newline, or newline followed 
		by carriage return) is converted to a simple newline. When the 
		opening long bracket is immediately followed by a newline, the 
		newline is not included in the string.
		
		As an example, in a system using ASCII 
		(in which 'a' is coded as 97, newline is 
		coded as 10, and '1' is coded as 49), the 
		five literal strings below 
		denote the same string:

		```
		a = 'alo\n123"'
		a = "alo\n123\""
		a = '\97lo\10\04923"'
		a = [[alo
		123"]]
		a = [==[
		alo
		123"]==]
		```
		Any byte in a literal string not explicitly affected by 
		the previous rules represents itself.
	*/
	inline std::string readStringLiteral(AnyInput auto& in,const char firstTypeChar)
	{
		in.skip();//skip firstTypeChar

		ParseNewlineState nlState = ParseNewlineState::NONE;

		std::string result;
		if (firstTypeChar == '"' || firstTypeChar == '\'')
		{
			while(true)
			{
				const char c = in.get();
				if (c == firstTypeChar) break;

				//No skip pre nl, cuz it will error anyway.
				if (manageNewlineState<false>(c, nlState, in))
				{
					throw UnexpectedCharacterError(
						LUACC_INVALID "Unfinished " LC_string ", line ended"
						+ errorLocStr(in)
					);
				}

				if (c == '\\')
				{
					const char next = in.get();
					switch (next)
					{
					case '\n':
					case '\r':
						//No skip pre nl, cuz loc is not visible in painting.
						manageNewlineState<false>(next, nlState, in);
						if (next == '\r')
						{// \r\n is treated as 1 char here
							if (in.peek() == '\n')
							{// Manual newline state handling
								in.skip();
								in.newLine();
								nlState = ParseNewlineState::NONE;
							}
						}
						result += '\n';
						break;

					case '\\': result += '\\'; break;
					case '\'': result += '\''; break;
					case '"': result += '"'; break;
					case 'a': result += '\a'; break;
					case 'b': result += '\b'; break;
					case 'f': result += '\f'; break;
					case 'n': result += '\n'; break;
					case 'r': result += '\r'; break;
					case 't': result += '\t'; break;
					case 'v': result += '\v'; break;
					case 'z':
					{
						while (true)
						{
							const char checkChar = in.peek();
							if (!isSpaceChar(checkChar))
								break;
							if (manageNewlineState<true>(checkChar, nlState, in))
								continue;

							in.skip();
						}
						break;
					}
					case 'x':
					{
						const char hex1 = in.get(), hex2 = in.get();
						result += (hexDigit2Num(hex1) << 4) | hexDigit2Num(hex2);
						break;
					}
					case 'u':
					{
						requireToken<false>(in, "{");

						uint32_t ch = 0;

						for (uint8_t i = 0; i < 9; i++)
						{
							if (i == 9 - 1)
							{
								requireToken<false>(in, "}");
								break;
							}
							const char escapeChar = in.get();
							if (escapeChar == '}')
								break;

							if (!isHexDigitChar(escapeChar))
							{
								throw UnexpectedCharacterError(std::format(
									"Expected a hex digit or " LUACC_SINGLE_STRING_INCOL(LUACC_BRACKET,"}}") " for unicode character, but found "
									LUACC_SINGLE_STRING("{}")
									"{}"
								, escapeChar, errorLocStr(in)));
							}
							ch <<= 4;
							ch |= hexDigit2Num(escapeChar);
						}
						if (ch > INT32_MAX)
						{
							throw UnicodeError(std::format(
								LC_Invalid " unicode character (more than " LUACC_NUM_COL("31") " bits!) "
								LUACC_NUMBER "{:x}" LUACC_DEFAULT
								"{}"
								, ch, errorLocStr(in))
							);
						}

						result += u32ToUtf8(ch);
						break;
					}
					case '0':
					case '1':
					case '2':
					case '3':
					case '4':
					case '5':
					case '6':
					case '7':
					case '8':
					case '9':
					{
						uint16_t num = next-'0';

						const char ch2 = in.peek();
						if (isDigitChar(ch2))
						{
							in.skip();
							num *= 10;
							num += ch2 - '0';


							const char ch3 = in.peek();
							if (isDigitChar(ch3))
							{
								in.skip();
								num *= 10;
								num += ch3 - '0';

								if (num > 0xFF)
								{
									throw UnicodeError(std::format(
										"Unicode " LC_not " allowed in " LUACC_SINGLE_STRING("\\ddd") "! "
										LUACC_NUMBER "{}" LUACC_DEFAULT
										"{}"
										, num, errorLocStr(in))
									);
								}
							}
						}

						result += (char)num;
						break;
					}
					default: 
						throw UnexpectedCharacterError(std::format(
							"Expected a escape sequence, but found "
							LUACC_SINGLE_STRING("{}")
							"{}"
						, next, errorLocStr(in)));
					}
				}
				else
					result += c;
			}
		}
		else if (firstTypeChar == '[')
		{
			size_t level = 0;
			while (in.peek() == '=')
			{
				level++;
				in.skip();
			}
			if constexpr (in.settings() & sluSyn)
			{
				if (level == 0)
					throw UnexpectedCharacterError(
						"Expected " LC_string 
						", to have atleast "
						LUACC_NUM_COL("1")
						" " LUACC_SINGLE_STRING("=") 
						" character between the "
						LUACC_SINGLE_STRING("[")
						" characters"
						+ errorLocStr(in)
					);
			}
			requireToken<false>(in, "[");

			bool skipNl = true;

			while (true)
			{
				const char c = in.get();
				//No skip pre nl, cuz it is not visible in painting.
				if (manageNewlineState<false>(c, nlState, in))
				{
					if (!skipNl)
						result += '\n';
					skipNl = false;//dont skip 2 newlines
				}
				if (c == '\n' || c == '\r')
					continue;
				skipNl = false;//dont skip newlines, after non-newline chars

				if (c == ']')
				{
					size_t closeLevel = 0;
					while (in.peek() == '=')
					{
						++closeLevel;
						in.skip();
					}
					if (in.peek() == ']' && closeLevel == level)
					{
						in.skip();
						break;
					}
					else
					{
						result += ']';
						result.append(closeLevel, '=');
					}
				}
				else
				{
					result += c;
				}
			}
		}
		return result;
	}
}