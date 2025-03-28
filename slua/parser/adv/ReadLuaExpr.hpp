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
#include <slua/parser/adv/ReadExpr.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/adv/ReadStringLiteral.hpp>
#include <slua/parser/adv/ReadNumeral.h>
#include <slua/parser/basic/ReadOperators.hpp>
#include <slua/parser/errors/CharErrors.h>

namespace sluaParse
{
	inline Expression readLuaExpr(AnyInput auto& in, const bool allowVarArg, const bool readBiOp = true)
	{
		/*
			nil | false | true | Numeral | LiteralString | ‘...’ | functiondef
			| prefixexp | tableconstructor | exp binop exp | unop exp
		*/

		const Position startPos = in.getLoc();

		bool isNilIntentional = false;
		Expression basicRes;
		basicRes.place = startPos;
		while (true)
		{
			const UnOpType uOp = readOptUnOp(in);
			if (uOp == UnOpType::NONE)break;
			basicRes.unOps.push_back(uOp);
		}

		skipSpace(in);

		const char firstChar = in.peek();
		switch (firstChar)
		{
		case 'n':
			if (checkReadTextToken(in, "nil"))
			{
				basicRes.data = ExprType::NIL();
				isNilIntentional = true;
				break;
			}
			break;
		case 'f':

			if (checkReadTextToken(in, "false")) { basicRes.data = ExprType::FALSE(); break; }
			if (checkReadTextToken(in, "function")) 
			{
				const Position place = in.getLoc();

				try
				{
					auto [fun, err] = readFuncBody(in);
					basicRes.data = ExprType::FUNCTION_DEF(std::move(fun));
					if (err)
					{

						in.handleError(std::format(
							"In lambda " LC_function " at {}",
							errorLocStr(in, place)
						));
					}
				}
				catch (const ParseError& e)
				{
					in.handleError(e.m);
					throw ErrorWhileContext(std::format(
						"In lambda " LC_function " at {}",
						errorLocStr(in, place)
					));
				}
				break; 
			}
			break;
		case 't':
			if (checkReadTextToken(in, "true")) { basicRes.data = ExprType::TRUE(); break; }
			break;
		case '.':
			if (checkReadToken(in, "..."))
			{
				if (!allowVarArg)
				{
					throw UnexpectedCharacterError(std::format(
						"Found varargs (" LUACC_SINGLE_STRING("...") ") "
						"outside of a vararg " LC_function " or the root " LC_function
						"{}"
						, errorLocStr(in)));
				}
				basicRes.data = ExprType::VARARGS();
				break;
			}
			[[fallthrough]];//handle as numeral instead (.0123, etc)
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
			basicRes.data = readNumeral(in,firstChar);
			break;
		case '"':
		case '\'':
		case '[':
			basicRes.data = ExprType::LITERAL_STRING(readStringLiteral(in, firstChar));
			break;
		case '{':
			basicRes.data = ExprType::TABLE_CONSTRUCTOR(readTableConstructor(in,allowVarArg));
			break;
		}

		if (!isNilIntentional && std::holds_alternative<ExprType::NIL>(basicRes.data)
			&&(firstChar=='(' || isValidNameStartChar(firstChar))
			)
		{//Prefix expr! or func-call

			basicRes.data = parsePrefixExprVar<ExprData,true>(in,allowVarArg, firstChar);
		}
		//check bin op

		if (!readBiOp)return basicRes;

		skipSpace(in);

		if(!in)
			return basicRes;//File ended

		const BinOpType firstBinOp = readOptBinOp(in);

		if (firstBinOp == BinOpType::NONE)
			return basicRes;

		ExprType::MULTI_OPERATION resData{};

		resData.first = std::make_unique<Expression>(std::move(basicRes));
		resData.extra.emplace_back(firstBinOp, readExpr(in,allowVarArg,false));

		while (true)
		{
			skipSpace(in);

			if (!in)
				break;//File ended

			const BinOpType binOp = readOptBinOp(in);

			if (binOp == BinOpType::NONE)
				break;

			resData.extra.emplace_back(binOp, readLuaExpr(in,allowVarArg,false));
		}
		Expression ret;
		ret.place = startPos;
		ret.data = std::move(resData);

		return ret;
	}

	inline ExpList readExpList(AnyInput auto& in, const bool allowVarArg)
	{
		/*
			explist ::= exp {‘,’ exp}
		*/
		ExpList ret{};
		ret.emplace_back(readExpr(in,allowVarArg));

		while (checkReadToken(in, ","))
		{
			ret.emplace_back(readExpr(in,allowVarArg));
		}
		return ret;
	}
}