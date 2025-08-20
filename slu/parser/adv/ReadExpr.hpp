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

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/ReadExprBase.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadStringLiteral.hpp>
#include <slu/parser/adv/ReadNumeral.hpp>
#include <slu/parser/basic/ReadOperators.hpp>
#include <slu/parser/errors/CharErrors.h>
#include <slu/parser/adv/ReadTraitExpr.hpp>
#include <slu/parser/adv/ReadTypeExpr.hpp>

namespace slu::parse
{
	template<AnyInput In>
	inline Lifetime readLifetime(In& in)
	{
		Lifetime res;
		do {
			in.skip();
			res.emplace_back(in.genData.resolveName(readName(in)));
			skipSpace(in);
		} while (in.peek() == '/');

		return res;
	}
	template<AnyInput In>
	inline UnOpItem readToRefLifetimes(In& in, UnOpType uOp)
	{
		skipSpace(in);
		if (in.peek() == '/')
		{// lifetime parsing, check for 'mut'

			Lifetime res = readLifetime(in);

			if (checkReadTextToken(in, "mut"))
				uOp = UnOpType::TO_REF_MUT;
			else if (checkReadTextToken(in, "const"))
				uOp = UnOpType::TO_REF_CONST;
			else if (checkReadTextToken(in, "share"))
				uOp = UnOpType::TO_REF_SHARE;

			return { std::move(res),uOp };
		}
		return { .type = uOp };
	}

	template<AnyInput In>
	inline void handleOpenRange(In& in, Expr<In>& basicRes)
	{
		if (basicRes.unOps.empty())return;

		if (basicRes.unOps.back().type == UnOpType::RANGE_BEFORE)
		{
			basicRes.unOps.erase(basicRes.unOps.end());
			basicRes.data = ExprType::OpenRange();
		}
	}

	template<bool IS_BASIC=false,bool FOR_PAT=false,AnyInput In>
	inline Expr<In> readExpr(In& in, const bool allowVarArg, const bool readBiOp = true)
	{
		/*
			nil | false | true | Numeral | LiteralString | ‘...’ | functiondef
			| prefixexp | tableconstructor | exp binop exp | unop exp
		*/
		const Position startPos = in.getLoc();

		bool isNilIntentional = false;
		Expr<In> basicRes;
		basicRes.place = startPos;


		while (true)
		{
			const UnOpType uOp = readOptUnOp(in);
			if (uOp == UnOpType::NONE)break;
			if constexpr (In::settings()&sluSyn)
			{
				if (uOp == UnOpType::TO_REF)
				{
					basicRes.unOps.push_back(readToRefLifetimes(in, uOp));
					continue;
				}
			}
			basicRes.unOps.push_back({.type= uOp });
		}
		skipSpace(in);

		const char firstChar = in.peek();
		switch (firstChar)
		{
		default:
			break;
		case '_':
			if constexpr (in.settings() & sluSyn)
			{
				if(!in.isOob(1) && !isValidNameChar(in.peekAt(1)))
				{
					in.skip();
					basicRes.data = ExprType::Infer{};
					break;
				}
			}
			[[fallthrough]];
		case ')':
		case '}':
		case ']':
		case ';':
		case ',':
		case '>':
		case '<':
		case '=':
		case '+':
		case '%':
		case '^':
		//case '&': //ref op
		//case '*': //Maybe a deref?
		//case '!':
		case '|'://todo: handle as lambda
		case '#':
			if constexpr (in.settings() & sluSyn)
			{
				handleOpenRange(in,basicRes);
			}
			break;
		case '~':
			if constexpr (in.settings() & sluSyn)
			{
				if (in.peekAt(1) == '~') // '~~'
				{
					requireToken(in, "~~");
					basicRes.data = ExprType::Err{ std::make_unique<ExprV<true>>(readExpr<IS_BASIC>(in,allowVarArg)) };
					break;
				}
			}
			break;
		case '/':
			if constexpr (in.settings() & sluSyn)
			{
				basicRes.data = readLifetime(in);
				break;
			}
			break;
		case 'd':
			if constexpr (in.settings() & sluSyn)
			{
				if (checkReadTextToken(in, "dyn"))
				{
					basicRes.data = ExprType::Dyn{readTraitExpr(in)};
					break;
				}
			}
			break;
		case 'i':
			if constexpr (in.settings() & sluSyn)
			{
				if (checkReadTextToken(in, "impl"))
				{
					basicRes.data = ExprType::Impl{readTraitExpr(in)};
					break;
				}
				if (checkReadTextToken(in, "if"))
				{
					//TODO: isloop
					basicRes.data = readIfCond<false, true, IS_BASIC>(
						in, allowVarArg
					);
					break;
				}
			}
			break;
		case 's'://safe fn
			if constexpr (in.settings() & sluSyn)
			{
				if (!checkReadTextToken(in, "safe"))
				{
					if (checkReadTextToken(in, "struct"))
					{
						requireToken(in, "{");
						basicRes.data = ExprType::Struct(readTable<false>(in, false));
					}
					break;
				}
				requireToken(in, "fn");
				basicRes.data = readFnType<IS_BASIC>(in, OptSafety::SAFE);
			}
			break;
		case 'u'://unsafe fn
			if constexpr (in.settings() & sluSyn)
			{
				if (!checkReadTextToken(in, "unsafe"))
				{
					if (checkReadTextToken(in, "union"))
					{
						requireToken(in, "{");
						basicRes.data = ExprType::Union(readTable<false>(in, false));
					}
					break;
				}
				requireToken(in, "fn");
				basicRes.data = readFnType<IS_BASIC>(in, OptSafety::UNSAFE);
			}
			break;
		case 'f':
			if constexpr(in.settings() & sluSyn)
			{
				if (checkReadTextToken(in, "fn"))
				{
					basicRes.data = readFnType<IS_BASIC>(in, OptSafety::DEFAULT);
					break;
				}
			} else {
				if (checkReadTextToken(in, "false")) 
				{
					basicRes.data = ExprType::False();
					break;
				}
			}
			if (checkReadTextToken(in, "function")) 
			{
				const Position place = in.getLoc();

				try
				{
					auto [fun, err] = readFuncBody(in,std::nullopt);
					ezmatch(std::move(fun))(
					varcase(Function<In>&&)
					{
						basicRes.data = ExprType::Function<In>(std::move(var));
						if (err)
						{

							in.handleError(std::format(
								"In lambda " LC_function " at {}",
								errorLocStr(in, place)
							));
						}
					},
					varcase(FunctionInfo<In>&&)
					{
						throw UnexpectedCharacterError(std::format(
							"Expected a " 
							LC_function 
							" block for lambda, at"
							"{}", errorLocStr(in, place)
						));
					}
					);
				}
				catch (const ParseError& e)
				{
					in.handleError(e.m);
					throw ErrorWhileContext(std::format(
						"In lambda " LC_function " at{}",
						errorLocStr(in, place)
					));
				}
				break; 
			}
			break;
		case 'n':
			if constexpr (In::settings() & sluSyn)break;

			if (checkReadTextToken(in, "nil"))
			{
				basicRes.data = ExprType::Nil{};
				isNilIntentional = true;
				break;
			}
			break;
		case 't':
			if constexpr (!(In::settings() & sluSyn))
			{
				if (checkReadTextToken(in, "true")) 
				{
					basicRes.data = ExprType::True(); 
					break; 
				}
			}
			break;
		case '.':
			if constexpr(!(In::settings() & sluSyn))
			{
				if (checkReadToken(in, "..."))
				{
					if (!allowVarArg)
					{
						in.handleError(std::format(
							"Found varargs (" LUACC_SINGLE_STRING("...") ") "
							"outside of a vararg " LC_function " or the root " LC_function
							"{}"
							, errorLocStr(in)));
					}
					basicRes.data = ExprType::VarArgs();
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
			basicRes.data = readNumeral<ExprData<In>>(in,firstChar);
			break;
		case '"':
		case '\'':
		case '[':
			if constexpr(In::settings()&sluSyn)
			{
				if (firstChar!='[' || in.peekAt(1) == '=')// [=....
					basicRes.data = ExprType::String(readStringLiteral(in, firstChar),in.getLoc());
				else
				{// must be a slicer [x]
					in.skip();

					basicRes.data = ExprType::Slice{
						std::make_unique<ExprV<true>>(
							readExpr(in, allowVarArg)
					)};
					requireToken(in, "]");
				}
			}
			else
				basicRes.data = ExprType::String(readStringLiteral(in, firstChar), in.getLoc());

			break;
		case '{':
			if constexpr (FOR_PAT)
			{
				handleOpenRange(in, basicRes);
				basicRes.data = ExprType::PatTypePrefix{};
				return basicRes;
			}
			else
				basicRes.data = ExprType::Table<In>(readTable(in,allowVarArg));
			break;
		}
		if (!isNilIntentional
			&& std::holds_alternative<ExprType::Nil>(basicRes.data))
		{//Prefix expr! or func-call

			bool maybeRootMp = false;
			if constexpr (In::settings() & sluSyn)
			{
				maybeRootMp = firstChar == ':';
			}

			if (!maybeRootMp && firstChar != '(' && !isValidNameStartChar(firstChar))
				throwExpectedExpr(in);

			if constexpr (FOR_PAT)
			{
				if (!maybeRootMp && firstChar != '(')
				{// check if unops are a type prefix
					size_t iterIdx = 1;//First was valid
					
					while (isValidNameChar(in.peekAt(iterIdx)))
						iterIdx++;
					// Lands on non valid idx
					iterIdx = weakSkipSpace(in, iterIdx);

					// => in = , }, BUT not ==
					const char nextChar = in.peekAt(iterIdx);
					if ((nextChar == '=' && in.peekAt(iterIdx + 1) != '=')
						|| nextChar == ',' || nextChar == '}'
						|| nextChar=='i'&&(in.peekAt(iterIdx + 1)=='n' && !isValidNameChar(in.peekAt(iterIdx + 2)))
						)
					{
						basicRes.data = ExprType::PatTypePrefix{};
						return basicRes;
					}
				}
			}

			basicRes.data = parsePrefixExprVar<ExprData<In>,true, IS_BASIC>(in,allowVarArg, firstChar);
		}
		if constexpr(in.settings() & sluSyn)
		{//Postfix op

			while (true)
			{
				const PostUnOpType uOp = readOptPostUnOp<true>(in);
				if (uOp == PostUnOpType::NONE)break;
				basicRes.postUnOps.push_back(uOp);
			}

			skipSpace(in);

			if (checkToken(in, ".."))
			{
				//binop or postunop?
				const size_t nextCh = weakSkipSpace(in, 2);
				const char nextChr = in.peekAt(nextCh);
				if (nextChr == '.')
				{
					const char dotChr = in.peekAt(nextCh + 1);
					if(dotChr<'0' && dotChr>'9')
					{//Is not number (.xxxx)
						in.skip(2);
						basicRes.postUnOps.push_back(PostUnOpType::RANGE_AFTER);
					}
				}
				else if (
					(nextChr >= 'a' && nextChr <= 'z')
					|| (nextChr >= 'A' && nextChr <= 'Z'))
				{
					if (peekName<NameCatagory::MP_START>(in, nextCh) == SIZE_MAX)
					{//Its reserved
						in.skip(2);
						basicRes.postUnOps.push_back(PostUnOpType::RANGE_AFTER);
					}
				}
				else if (// Not 0-9,_,",',$,[,{,(
					(nextChr < '0' || nextChr > '9')
					&& nextChr!='_'
					&& nextChr!='"'
					&& nextChr!='\''
					&& nextChr!='$'
					&& nextChr!='['
					&& nextChr!='{'
					&& nextChr!='('
				)
				{
					in.skip(2);
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

		ExprType::MultiOp<In> resData{};

		resData.first = std::make_unique<Expr<In>>(std::move(basicRes));
		resData.extra.emplace_back(firstBinOp, readExpr<IS_BASIC>(in,allowVarArg,false));

		while (true)
		{
			skipSpace(in);

			if (!in)
				break;//File ended

			const BinOpType binOp = readOptBinOp(in);

			if (binOp == BinOpType::NONE)
				break;

			resData.extra.emplace_back(binOp, readExpr<IS_BASIC>(in,allowVarArg,false));
		}
		Expr<In> ret;
		ret.place = startPos;
		ret.data = std::move(resData);

		return ret;
	}
}