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
		// NOTE: approximate!!

		Numeral ::= HexNum | DecNum


		DecNum ::= ((DigList ["." [DigList]]) | "." DigList) [DecExpSep [NumSign] DigList]

		HexNum ::= "0" HexSep ((HexDigList ["." [HexDigList]]) | "." HexDigList) [HexExpSep [NumSign] DigList]

		DecExpSep	::= "e" | "E"
		HexSep		::= "x" | "X"
		HexExpSep	::= "p" | "P"

		DigList		::= Digit {Digit}
		HexDigList	::= HexDigit {HexDigit}

		NumSign ::= "+" | "-"

		Digit		::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
		HexDigit	::= Digit | a | b | c | d | e | f | A | B | C | D | E | F
	*/

	inline ExprData readNumeralExtra(AnyInput auto& in, const bool hex, const bool decPart, const char firstChar)
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

            const bool digit = hex ? isHexDigitChar(c) : isDigitChar(c);

            if (digit)
            {
                number += in.get();
                requireExpCh = false;
            }
            else if (c == '.' && !hasDot && !hasExp)
            {
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
                    "Expected a digit or " LUACC_SINGLE_STRING("+") "/" LUACC_SINGLE_STRING("-") " for exponent of " LC_number
                    + errorLocStr(in));
            }
            else
                break;
        }

        if (requireExpCh && !in)
        {
            throw 333;
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
            if (hex)
            {
                number = "0x" + number;
            }
            return ExprType::NUMERAL(std::stod(number));
        }
        else
        {
            try
            {
                return ExprType::NUMERAL_I64(std::stoll(number, nullptr, hex ? 16 : 10));
            }
            catch (const std::exception&)
            {
                if (hex)
                {
                    number = "0x" + number;
                }
                return ExprType::NUMERAL(std::stod(number));
            }
        }
        }
        catch (const std::out_of_range&)
        {
            return ExprType::NUMERAL(INFINITY);//Always positive, since negative is handled by un-ops, in expressions
        }
        catch (const std::exception&)
        {
            throw UnexpectedCharacterError(
                LUACC_INVALID "Malformed " LC_number
                " "
                LUACC_START_SINGLE_STRING
                +number+
                LUACC_END_SINGLE_STRING
                + errorLocStr(in));
        }
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