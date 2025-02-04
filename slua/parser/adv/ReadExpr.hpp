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
#include <slua/parser/adv/ReadStringLiteral.hpp>
#include <slua/parser/adv/ReadNumeral.h>
#include <slua/parser/basic/ReadOperators.hpp>

namespace sluaParse
{
	inline void parseVarBase(AnyInput auto& in, const char firstChar, Var& varDataOut, bool& varDataNeedsSubThing)
	{
		if (firstChar == '(')
		{// Must be '(' exp ')'
			in.skip();
			BaseVarType::EXPR res(readExpr(in));
			requireToken(in, ")");
			varDataNeedsSubThing = true;
			varDataOut.base = std::move(res);
		}
		else
		{// Must be Name
			varDataOut.base = BaseVarType::NAME(readName(in));
		}
	}

	template<class T,bool FOR_EXPR>
	inline T parsePrefixExprVar(AnyInput auto& in, const char firstChar)
	{
		/*
			var ::= baseVar {subvar}

			baseVar ::= Name | ‘(’ exp ‘)’ subvar

			funcArgs ::=  [‘:’ Name] args
			subvar ::= {funcArgs} ‘[’ exp ‘]’ | {funcArgs} ‘.’ Name
		*/

		std::vector<Var> varData;
		std::vector<ArgFuncCall> funcCallData;// Current func call chain, empty->no chain
		bool varDataNeedsSubThing = false;

		varData.emplace_back();
		parseVarBase(in, firstChar, varData.back(), varDataNeedsSubThing);

		//This requires manual parsing, and stuff (at every step, complex code)
		while (true)
		{
			skipSpace(in);

			const char opType = in.peek();
			switch (opType)
			{
			case ',':// Varlist
				if constexpr (FOR_EXPR)
				{
					throw UnexpectedCharacterError(
						"Cant assign in expression, found "
						LUACC_SINGLE_STRING("=")
						+ errorLocStr(in));
				}
				else
				{
					if (!funcCallData.empty())
					{
						throw UnexpectedCharacterError(
							"Cant assign to " LC_function " call (Found in variable list)"
							+ errorLocStr(in));
					}
					if (varDataNeedsSubThing)
					{
						throw UnexpectedCharacterError(
							"Cant assign to expression, found "
							LUACC_SINGLE_STRING("=")
							+ errorLocStr(in));
					}
					skipSpace(in);
					varData.emplace_back();
					parseVarBase(in, in.peek(), varData.back(), varDataNeedsSubThing);
					break;
				}
			case '=':// Assign
			{
				if constexpr (FOR_EXPR)
				{
					throw UnexpectedCharacterError(
						"Cant assign in expression, found "
						LUACC_SINGLE_STRING("=")
						+ errorLocStr(in));
				}
				else
				{
					if (!funcCallData.empty())
					{
						throw UnexpectedCharacterError(
							"Cant assign to " LC_function " call, found "
							LUACC_SINGLE_STRING("=")
							+ errorLocStr(in));
					}
					if (varDataNeedsSubThing)
					{
						throw UnexpectedCharacterError(
							"Cant assign to expression, found "
							LUACC_SINGLE_STRING("=")
							+ errorLocStr(in));
					}
					in.skip();
					StatementType::ASSIGN res{};
					res.vars = std::move(varData);
					res.exprs = readExpList(in);
					return res;
				}
			}
			case ':'://Self funccall
			{
				in.skip();
				std::string name = readName(in);

				funcCallData.emplace_back(name, readArgs(in));
				break;
			}
			case '{':
			case '"':
			case '('://Funccall
				funcCallData.emplace_back("", readArgs(in));
				break;
			case '.':// Index
			{
				in.skip();

				SubVarType::NAME res{};
				res.funcCalls = std::move(funcCallData);// Move auto-clears it
				res.idx = readName(in);

				varDataNeedsSubThing = false;
				varData.back().sub.emplace_back(std::move(res));
				break;
			}
			case '[':// Arr-index
			{
				SubVarType::EXPR res{};
				res.funcCalls = std::move(funcCallData);// Move auto-clears it

				in.skip();
				res.idx = readExpr(in);
				requireToken(in, "]");

				varDataNeedsSubThing = false;
				varData.back().sub.emplace_back(std::move(res));
				break;
			}
			default:
			{
				_ASSERT(!varData.empty());
				if (varData.size() != 1)
				{
					throw UnexpectedCharacterError(std::format(
						"Expected multi-assignment, found "
						LUACC_START_SINGLE_STRING "{}" LUACC_END_SINGLE_STRING
						"{}"
					, opType, errorLocStr(in)));
				}
				if (funcCallData.empty())
				{
					if constexpr (FOR_EXPR)
					{
						if (varDataNeedsSubThing)
						{
							LimPrefixExprType::EXPR res;
							res.v = std::move(std::get<BaseVarType::EXPR>(varData.back().base).start);
							return std::make_unique<LimPrefixExpr>(std::move(res));
						}
						return std::make_unique<LimPrefixExpr>(LimPrefixExprType::VAR(std::move(varData.back())));
					}
					else
					{
						if (varDataNeedsSubThing)
						{
							throw UnexpectedCharacterError(
								"Raw expressions are not allowed, expected assignment or " LC_function " call"
								+ errorLocStr(in));
						}
						throw UnexpectedCharacterError(std::format(
							"Expected assignment or " LC_function " call, found "
							LUACC_START_SINGLE_STRING "{}" LUACC_END_SINGLE_STRING
							"{}"
						, opType, errorLocStr(in)));
					}
				}
				if (varDataNeedsSubThing)
				{
					BaseVarType::EXPR& bVarExpr = std::get<BaseVarType::EXPR>(varData.back().base);
					auto limP = LimPrefixExprType::EXPR(std::move(bVarExpr.start));
					return FuncCall(std::make_unique<LimPrefixExpr>(std::move(limP)), std::move(funcCallData));
				}
				auto limP = LimPrefixExprType::VAR(std::move(varData.back()));
				return FuncCall(std::make_unique<LimPrefixExpr>(std::move(limP)), std::move(funcCallData));
			}
			}
		}
	}

	inline Expression readExpr(AnyInput auto& in, const bool readBiOp = true)
	{
		/*
			nil | false | true | Numeral | LiteralString | ‘...’ | functiondef
			| prefixexp | tableconstructor | exp binop exp | unop exp
		*/

		const Position startPos = in.getLoc();

		bool isNilIntentional = false;
		Expression basicRes;
		basicRes.place = startPos;
		basicRes.unOp = readOptUnOp(in);

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
			if (checkReadTextToken(in, "function")) { basicRes.data = ExprType::FUNCTION_DEF(readFuncBody(in)); break; }
			break;
		case 't':
			if (checkReadTextToken(in, "true")) { basicRes.data = ExprType::TRUE(); break; }
			break;
		case '.':
			if (checkReadToken(in, "..."))
			{
				basicRes.data = ExprType::VARARGS();
				break;
			}
			//break;
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
			basicRes.data = ExprType::TABLE_CONSTRUCTOR(readTableConstructor(in));
			break;
		}

		if (!isNilIntentional && std::holds_alternative<ExprType::NIL>(basicRes.data)
			&&(firstChar=='(' || isValidNameStartChar(firstChar))
			)
		{//Prefix expr! or func-call

			basicRes.data = parsePrefixExprVar<ExprData,true>(in, firstChar);
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
		resData.extra.emplace_back(firstBinOp, readExpr(in,false));

		while (true)
		{
			skipSpace(in);

			if (!in)
				break;//File ended

			const BinOpType binOp = readOptBinOp(in);

			if (binOp == BinOpType::NONE)
				break;

			resData.extra.emplace_back(binOp, readExpr(in,false));
		}
		Expression ret;
		ret.place = startPos;
		ret.data = std::move(resData);

		return ret;
	}

	inline ExpList readExpList(AnyInput auto& in)
	{
		/*
			explist ::= exp {‘,’ exp}
		*/
		ExpList ret{};
		ret.emplace_back(readExpr(in));

		while (checkReadToken(in, ","))
		{
			ret.emplace_back(readExpr(in));
		}
		return ret;
	}
}