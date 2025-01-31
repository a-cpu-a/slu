/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/basic/CharInfo.hpp>

namespace sluaParse
{
	/*

		// NOTE: NO WHITESPACE BETWEEN CHARS!!!

		Numeral ::= HexNum | DecNum


		DecNum ::= ((DigList ["." [DigList]]) | "." DigList) [DecExpSep [NumSign] DigList]

		HexNum ::= "0" HexSep ((HexDigList ["." [HexDigList]]) | "." HexDigList) [HexExpSep [NumSign] DigList]

		DecExpSep	::= "e" | "E"
		HexSep		::= "x" | "X"
		HexExpSep	::= "p" | "P"

		DigList ::= Digit {Digit}
		HexDigList ::= HexDigit {HexDigit}

		NumSign ::= "+" | "-"

		Digit ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
		HexDigit ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | a | b | c | d | e | f | A | B | C | D | E | F
	*/

	inline ExprData readNumeralExtra(AnyInput auto& in, const bool hex, const bool decPart, const char firstChar)
	{
		return {};
	}

	inline ExprData readNumeral(AnyInput auto& in, const char firstChar)
	{
		in.skip();


		if (firstChar == '.')
		{
			//must be non-hex, float(or must it..?)
			return readNumeralExtra(in, false, true, firstChar);
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
		return readNumeralExtra(in,hex,false,firstChar);
	}
}