/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>
#include <memory>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/ReadExprBase.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/adv/ReadStringLiteral.hpp>
#include <slua/parser/adv/ReadNumeral.hpp>
#include <slua/parser/basic/ReadOperators.hpp>
#include <slua/parser/errors/CharErrors.h>

namespace sluaParse
{
	template<AnyInput In>
	inline Expression<In> readExpr(In& in, const bool allowVarArg, const bool readBiOp = true)
	{
		/*
			nil | false | true | Numeral | LiteralString | ‘...’ | functiondef
			| prefixexp | tableconstructor | exp binop exp | unop exp
		*/

		const Position startPos = in.getLoc();

		bool isNilIntentional = false;
		Expression<In> basicRes;
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
					basicRes.data = ExprType::FUNCTION_DEF<In>(std::move(fun));
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
			if constexpr(!(in.settings() & sluaSyn))
			{
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
			if constexpr(in.settings()&sluaSyn)
			{
				if (in.peekAt(1) == '=')// [=....
					basicRes.data = ExprType::LITERAL_STRING(readStringLiteral(in, firstChar));
				else
				{// must be a array constructor
					in.skip();
					Expression<In> firstItem = readExpr(in, allowVarArg);
					if (in.peek() == ';')
					{//[x;y]
						in.skip();
						ExprType::ARRAY_CONSTRUCTOR<In> res;
						res.val = std::make_unique<Expression<In>>(std::move(firstItem));
						res.size = std::make_unique<Expression<In>>(readExpr(in, allowVarArg));
						requireToken(in, "]");
						basicRes.data = std::move(res);
					}
					else
					{//[x {, ...} ]
						ExprType::ARRAY_CONSTRUCTOR_LIST<In> res;
						res.values.emplace_back(std::move(firstItem));
						while (in.peek() == ',')
						{
							res.values.emplace_back(readExpr(in, allowVarArg));
						}
						requireToken(in, "]");
						basicRes.data = std::move(res);
					}
				}
			}
			else
				basicRes.data = ExprType::LITERAL_STRING(readStringLiteral(in, firstChar));

			break;
		case '{':
			basicRes.data = ExprType::TABLE_CONSTRUCTOR<In>(readTableConstructor(in,allowVarArg));
			break;
		}

		if (!isNilIntentional && std::holds_alternative<ExprType::NIL>(basicRes.data)
			&&(firstChar=='(' || isValidNameStartChar(firstChar))
			)
		{//Prefix expr! or func-call

			basicRes.data = parsePrefixExprVar<ExprData<In>,true>(in,allowVarArg, firstChar);
		}
		if constexpr(in.settings() & sluaSyn)
		{//Postfix op
			while(checkReadToken(in,"?"))
				basicRes.postUnOps.push_back(PostUnOpType::PROPOGATE_ERR);
			skipSpace(in);

			if (checkToken(in, "..."))
			{
				//binop or postunop?
				const size_t nextCh = weakSkipSpace(in, 3);
				const char nextChr = in.peekAt(nextCh);
				if (nextChr == '.')
				{
					if(in.peekAt(nextCh+1)=='.')
					{//Is '..' or '...', etc
						in.skip(nextCh);
						basicRes.postUnOps.push_back(PostUnOpType::RANGE_AFTER);
					}
				}
				else if (
					(nextChr >= 'a' && nextChr <= 'z')
					&& (nextChr >= 'A' && nextChr <= 'Z'))
				{
					if (peekName<true>(in, nextCh) == SIZE_MAX)
					{//Its reserved
						in.skip(nextCh);
						basicRes.postUnOps.push_back(PostUnOpType::RANGE_AFTER);
					}
				}
				else if (// Not 0-9,_,",',$,[,{,(
					(nextChr < '0' && nextChr > '9')
					&& nextChr!='_'
					&& nextChr!='"'
					&& nextChr!='\''
					&& nextChr!='$'
					&& nextChr!='['
					&& nextChr!='{'
					&& nextChr!='('
				)
				{
					in.skip(nextCh);
					basicRes.postUnOps.push_back(PostUnOpType::RANGE_AFTER);
				}
			}
		}

		//check bin op
		if (!readBiOp)return basicRes;

		skipSpace(in);

		if(!in)
			return basicRes;//File ended

		const BinOpType firstBinOp = readOptBinOp(in);

		if (firstBinOp == BinOpType::NONE)
			return basicRes;

		ExprType::MULTI_OPERATION<In> resData{};

		resData.first = std::make_unique<Expression<In>>(std::move(basicRes));
		resData.extra.emplace_back(firstBinOp, readExpr(in,allowVarArg,false));

		while (true)
		{
			skipSpace(in);

			if (!in)
				break;//File ended

			const BinOpType binOp = readOptBinOp(in);

			if (binOp == BinOpType::NONE)
				break;

			resData.extra.emplace_back(binOp, readExpr(in,allowVarArg,false));
		}
		Expression<In> ret;
		ret.place = startPos;
		ret.data = std::move(resData);

		return ret;
	}
}