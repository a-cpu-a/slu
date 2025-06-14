﻿/*
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
#include <slu/parser/basic/CharInfo.hpp>

namespace slu::parse
{
	inline constexpr uint64_t LAST_DIGIT_HEX = 0xFULL << (64 - 4);

	constexpr int64_t parseHexInt(AnyInput auto& in, const std::string& str) {

		int64_t result = 0;

		for (const char c : str)
		{
			if constexpr (in.settings() & noIntOverflow)
			{
				if (((uint64_t)result & LAST_DIGIT_HEX) != 0)//does it have anything?
				{// Yes, exit now!
					reportIntTooBig(in, str);
					return result;
				}
			}

			result <<= 4;// bits per digit
			result |= hexDigit2Num(c);
		}

		return result;
	}

	constexpr ExprType::NUMERAL_U128 parseHexU128(AnyInput auto& in, const std::string& str) {

		ExprType::NUMERAL_U128 result;

		for (const char c : str)
		{
			if constexpr (in.settings() & noIntOverflow)
			{
				if ((result.hi & LAST_DIGIT_HEX)!=0)//does it have anything?
				{// Yes, exit now!
					reportIntTooBig(in, str);
					return result;
				}
			}
			result.hi <<= 4;
			result.hi |= (result.lo & LAST_DIGIT_HEX)>>60;
			result.lo <<= 4;// bits per digit

			result.lo |= hexDigit2Num(c);
		}

		return result;
	}
	constexpr ExprType::NUMERAL_U128 parseU128(AnyInput auto& in, const std::string& str) {

		ExprType::NUMERAL_U128 result;

		for (const char c : str)
		{
			uint8_t digit = (c - '0');


			if constexpr (in.settings() & noIntOverflow)
			{
				if ((
						//mul
						result.hi > (UINT64_MAX / 10)
						|| (result.hi == (UINT64_MAX / 10)
						&& result.lo > (UINT64_MAX - result.lo * 10) / 10)

					) || (
						//add
						result.hi > (UINT64_MAX - digit) / 10
						|| (result.lo > (UINT64_MAX - digit))
					)
				)
				{
					reportIntTooBig(in, str);
					return result;
				}
			}

			//multiply by 10 aka 0b1010
			result = result.shift(1) + result.shift(3);
			//atleast 0-9 wont overflow.
			result.lo += digit;
		}

		return result;
	}

	template<bool pad=false>
	constexpr std::string u64ToStr(const uint64_t v)
	{
		std::string res;
		for (size_t i = 0; i < 16; i++)
		{
			const uint64_t va = uint64_t(v) >> (60 - 4 * i);

			if constexpr (!pad)
			{
				if (va == 0)
					continue;
			}
			const uint8_t c = va & 0xF;
			if (c <= 9)
				res += ('0' + c);
			else
				res += ('A' + (c - 10));
		}
		return res;
	}

	constexpr std::string u128ToStr(const uint64_t lo, const uint64_t hi)
	{
		std::string res = "0x";
		if(hi!=0)
		{
			res += u64ToStr(hi);
			res += u64ToStr<true>(lo);
		}
		else if (lo != 0)
			res += u64ToStr(lo);
		else
			res += '0';
		return res;
	}

	/*

		// NOTE: NO WHITESPACE BETWEEN CHARS!!!
		// NOTE: approximate!!

		Numeral ::= HexNum | DecNum


		DecNum ::= ((DigList ["." [DigList]]) | "." DigList) [DecExpSep [NumSign] DigList] ["_" Name]

		HexNum ::= "0" HexSep ((HexDigList ["." [HexDigList]]) | "." HexDigList) [HexExpSep [NumSign] DigList] ["_" Name]

		DecExpSep	::= "e" | "E"
		HexSep		::= "x" | "X"
		HexExpSep	::= "p" | "P"

		DigList		::= Digit {Digit} [{Digit | "_"} Digit]
		HexDigList	::= HexDigit {HexDigit} [{Digit | "_"} Digit]

		NumSign ::= "+" | "-"

		Digit		::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
		HexDigit	::= Digit | a | b | c | d | e | f | A | B | C | D | E | F


	*/

	//TODO: parse types
	template<class DataT, bool ALLOW_FLOAT, bool allowSluType = true, AnyInput In>
	inline DataT readNumeralExtra(In& in, const bool hex, const bool decPart, const char firstChar)
	{
		std::string number;
		number += firstChar;
		bool hasDot = decPart;
		bool hasExp = false;
		bool isFloat = decPart;

		bool requireExpCh = false;

		const char expChar = hex ? 'p' : 'e';

		while (in)
		{
			const char c = in.peek();

			if constexpr (in.settings() & numberSpacing)
			{
				if (c == '_')
				{
					in.skip();
					continue;
				}
			}

			const bool digit = (hex && !hasExp) ? isHexDigitChar(c) : isDigitChar(c);

			if (digit)
			{
				number += in.get();
				requireExpCh = false;
			}
			else if (c == '.' && !hasDot && !hasExp)
			{
				if constexpr (In::settings()&sluSyn)
				{
					if (!in.isOob(1) && in.peekAt(1) == '.')
						break;
				}
				if constexpr (!ALLOW_FLOAT)
					break;

				hasDot = true;
				isFloat = true;
				number += in.get();
			}
			else if (isLowerCaseEqual(c, expChar) && !hasExp)
			{
				hasExp = true;
				isFloat = true;
				number += in.get();
				if (in.peek() == '+' || in.peek() == '-')
				{
					number += in.get();
				}
				requireExpCh = true;
			}
			else if (requireExpCh)
			{
				throw UnexpectedCharacterError(
					"Expected a digit or " 
					LUACC_SINGLE_STRING("+") "/" LUACC_SINGLE_STRING("-")
					" for exponent of " LC_number
					+ errorLocStr(in));
			}
			else
				break;
		}

		if (requireExpCh && !in)
		{
			throw UnexpectedCharacterError(
				"Expected a digit or "
				LUACC_SINGLE_STRING("+") "/" LUACC_SINGLE_STRING("-")
				" for exponent of " LC_number ", but file ended"
				+ errorLocStr(in));
		}

		if (in && isValidNameStartChar(in.peek()))
		{
			throw UnexpectedCharacterError(
				"Found a text character after a " LC_number
				+ errorLocStr(in));
		}
		try {
		if (isFloat)
		{
			if constexpr (ALLOW_FLOAT)
			{
				if (hex)
				{
					number = "0x" + number;
				}
				return ExprType::NUMERAL(std::stod(number));
			}
			throwUnexpectedFloat(in, number);
		}
		else
		{
			if constexpr (in.settings() & sluSyn)
			{
				ExprType::NUMERAL_U128 n128 = hex
					? parseHexU128(in, number)
					: parseU128(in, number);

				if (n128.hi >> 63)
					return n128;//I128 would be negative

				if (n128.hi != 0)
					return ExprType::NUMERAL_I128(n128);

				if (n128.lo >> 63)//I64 would be negative
					return ExprType::NUMERAL_U64(n128.lo);

				return ExprType::NUMERAL_I64(n128.lo);
			}

			if (hex)
				return ExprType::NUMERAL_I64(parseHexInt(in, number));
			try
			{
				return ExprType::NUMERAL_I64(std::stoll(number, nullptr, 10));
			}
			catch (...)
			{
				if constexpr (in.settings() & noIntOverflow)
					reportIntTooBig(in, number);
				if constexpr(ALLOW_FLOAT)
					return ExprType::NUMERAL(std::stod(number));
				throwUnexpectedFloat(in, number);
			}
		}
		}
		catch (const std::out_of_range&)
		{
			if constexpr (ALLOW_FLOAT)
				return ExprType::NUMERAL(INFINITY);//Always positive, since negative is handled by un-ops, in expressions
			throwUnexpectedFloat(in, number);
		}
		catch (const std::exception&)
		{
			throw UnexpectedCharacterError(std::format(
				LUACC_INVALID "Malformed " LC_number
				" " LUACC_SINGLE_STRING("{}")
				"{}", number, errorLocStr(in)));
		}
		throw InternalError("Failed to return from readNumeralExtra");
	}

	template<class DataT,bool ALLOW_FLOAT = true,bool allowSluType=true, AnyInput In>
	inline DataT readNumeral(In& in, const char firstChar)
	{
		in.skip();

		if (firstChar == '.')
		{
			//must be non-hex, float(or must it..?)
			return readNumeralExtra<DataT, ALLOW_FLOAT, allowSluType>(in, false, true, firstChar);
		}
		if (!in)
		{//The end of the stream!
			return ExprType::NUMERAL_I64(firstChar-'0');// turn char into int
		}
		bool hex = false;

		if (firstChar == '0' && toLowerCase(in.peek()) == 'x')
		{
			hex = true;
			in.skip();//skip 'x'
		}
		return readNumeralExtra<DataT, ALLOW_FLOAT, allowSluType>(in,hex,false,firstChar);
	}
}