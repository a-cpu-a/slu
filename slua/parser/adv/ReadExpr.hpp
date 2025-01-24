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
#include <slua/parser/basic/ReadOperators.hpp>

namespace sluaParse
{
	inline Expression readExpr(AnyInput auto& in)
	{
		/*
			nil | false | true | Numeral | LiteralString | ‘...’ | functiondef
			| prefixexp | tableconstructor | exp binop exp | unop exp
		*/

		Expression res;
		res.place = in.getLoc();

		res.unOp = readOptUnOp(in);

		skipSpace(in);

		switch (in.peek())
		{
		case 'n':
			if (checkReadTextToken(in, "nil"))
			{
				res.data = ExprType::NIL();
				break;
			}
			break;
		case 'f':

			if (checkReadTextToken(in, "false")) { res.data = ExprType::FALSE(); break; }
			if (checkReadTextToken(in, "function")) { res.data = ExprType::FUNCTION_DEF(readFuncBody(in)); break; }
			break;
		case 't':
			if (checkReadTextToken(in, "true")) { res.data = ExprType::TRUE(); break; }
			break;
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
			//TODO: numeral

			break;
		case '"':
		case '\'':
		case '[':
			//TODO: literal?
			break;
		case '.':
			if (checkReadToken(in, "..."))
			{
				res.data = ExprType::VARARGS();
				break;
			}
			break;
		case '(':
			//TODO: prefixexp
			break;
		case '{':
			//TODO: tableconstructor
			break;
		}

		//check bin op


		const BinOpType binOp = readOptBinOp(in);

		if (binOp == BinOpType::NONE)
			return res;
	}

	inline ExpList readExpList(AnyInput auto& in)
	{
		/*
			explist ::= exp {‘,’ exp}
		*/
		ExpList ret{};
		ret.push_back(readExpr(in));

		while (checkReadToken(in, ","))
		{
			ret.push_back(readExpr(in));
		}
		return ret;
	}
}