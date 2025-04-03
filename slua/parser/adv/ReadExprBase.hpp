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
#include <slua/parser/errors/CharErrors.h>

namespace sluaParse
{
	template<AnyInput In>
	inline ModPath parseModPath(In& in,const std::string& start)
	{
		ModPath mp = { start };
		skipSpace(in);
		while (checkToken(in, "::") && in.peekAt(2) != ':')
		{
			in.skip(2);//skip '::'
			skipSpace(in);
			mp.push_back(readName(in));
		}
		return mp;
	}

	template<AnyInput In>
	inline void parseVarBase(In& in, const bool allowVarArg, const char firstChar, Var<In>& varDataOut, bool& varDataNeedsSubThing)
	{
		if (firstChar == '(')
		{// Must be '(' exp ')'
			in.skip();
			BaseVarType::EXPR<In> res(readExpr(in,allowVarArg));
			requireToken(in, ")");
			varDataNeedsSubThing = true;
			varDataOut.base = std::move(res);
		}
		else
		{// Must be Name, ... or mod path

			//Lua doesnt reserve mp_start names, so doesnt matter
			std::string start = readName<true>(in);

			if constexpr (in.settings() & sluaSyn)
			{
				BaseVarType::MOD_PATH mp = parseModPath(in, start);

				if (mp.size() != 1)
					varDataOut.base = mp;
				return;
			}
			varDataOut.base = BaseVarType::NAME(start);
		}
	}

	template<class T,bool FOR_EXPR, AnyInput In>
	inline T returnPrefixExprVar(In& in, std::vector<Var<In>>& varData, std::vector<ArgFuncCall<In>>& funcCallData,const bool varDataNeedsSubThing,const char opTypeCh)
	{
		char opType[4] = "EOS";

		//Turn opType into "EOS", or opTypeCh
		if (opTypeCh != 0)
		{
			opType[0] = opTypeCh;
			opType[1] = 0;
		}

		_ASSERT(!varData.empty());
		if (varData.size() != 1)
		{
			if constexpr (FOR_EXPR)
				throwVarlistInExpr(in);

			throw UnexpectedCharacterError(std::format(
				"Expected multi-assignment, since there is a list of variables, but found "
				LUACC_SINGLE_STRING("{}")
				"{}"
				, opType, errorLocStr(in)));
		}
		if (funcCallData.empty())
		{
			if constexpr (FOR_EXPR)
			{
				if (varDataNeedsSubThing)
				{
					LimPrefixExprType::EXPR<In> res;
					res.v = std::move(std::get<BaseVarType::EXPR<In>>(varData.back().base).start);
					return std::make_unique<LimPrefixExpr<In>>(std::move(res));
				}
				return std::make_unique<LimPrefixExpr<In>>(LimPrefixExprType::VAR<In>(std::move(varData.back())));
			}
			else
			{
				if (varDataNeedsSubThing)
					throwRawExpr(in);

				throw UnexpectedCharacterError(std::format(
					"Expected assignment or " LC_function " call, found "
					LUACC_SINGLE_STRING("{}")
					"{}"
					, opType, errorLocStr(in)));
			}
		}
		if (varDataNeedsSubThing)
		{
			BaseVarType::EXPR<In>& bVarExpr = std::get<BaseVarType::EXPR<In>>(varData.back().base);
			auto limP = LimPrefixExprType::EXPR<In>(std::move(bVarExpr.start));
			return FuncCall<In>(std::make_unique<LimPrefixExpr<In>>(std::move(limP)), std::move(funcCallData));
		}
		auto limP = LimPrefixExprType::VAR<In>(std::move(varData.back()));
		return FuncCall<In>(std::make_unique<LimPrefixExpr<In>>(std::move(limP)), std::move(funcCallData));
	}
	template<class T,bool FOR_EXPR, AnyInput In>
	inline T parsePrefixExprVar(In& in, const bool allowVarArg, const char firstChar)
	{
		/*
			var ::= baseVar {subvar}

			baseVar ::= Name | ‘(’ exp ‘)’ subvar

			funcArgs ::=  [‘:’ Name] args
			subvar ::= {funcArgs} ‘[’ exp ‘]’ | {funcArgs} ‘.’ Name
		*/

		std::vector<Var<In>> varData;
		std::vector<ArgFuncCall<In>> funcCallData;// Current func call chain, empty->no chain
		bool varDataNeedsSubThing = false;

		varData.emplace_back();
		parseVarBase(in,allowVarArg, firstChar, varData.back(), varDataNeedsSubThing);

		char opType;

		//This requires manual parsing, and stuff (at every step, complex code)
		while (true)
		{
			const bool skipped = skipSpace(in);

			if (!in)
				return returnPrefixExprVar<T,FOR_EXPR>(in,varData, funcCallData, varDataNeedsSubThing,0);

			opType = in.peek();
			switch (opType)
			{
			case ',':// Varlist
				if constexpr (FOR_EXPR)
					goto exit;
				else
				{
					if (!funcCallData.empty())
						throwFuncCallInVarList(in);
					if (varDataNeedsSubThing)
						throwExprInVarList(in);

					in.skip();//skip comma
					skipSpace(in);
					varData.emplace_back();
					parseVarBase(in,allowVarArg, in.peek(), varData.back(), varDataNeedsSubThing);
					break;
				}
			default:
				goto exit;
			case '=':// Assign
			{
				if constexpr (FOR_EXPR)
					goto exit;
				else
				{
					if (!funcCallData.empty())
						throwFuncCallAssignment(in);
					if (varDataNeedsSubThing)
						throwExprAssignment(in);

					in.skip();//skip eq
					StatementType::ASSIGN<In> res{};
					res.vars = std::move(varData);
					res.exprs = readExpList(in,allowVarArg);
					return res;
				}
			}
			case ':'://Self funccall
			{
				if (in.peekAt(1) == ':') //is label / '::'
					goto exit;
				in.skip();//skip colon
				std::string name = readName(in);

				const bool skippedAfterName = skipSpace(in);

				if constexpr (in.settings() & spacedFuncCallStrForm)
				{
					if (!skippedAfterName)
						throwSpaceMissingBeforeString(in);
				}

				funcCallData.emplace_back(name, readArgs(in,allowVarArg));
				break;
			}
			case '"':
			case '\'':
				if constexpr (in.settings() & spacedFuncCallStrForm)
				{
					if (!skipped)
						throwSpaceMissingBeforeString(in);
				}
				[[fallthrough]];
			case '{':
			case '('://Funccall
				funcCallData.emplace_back("", readArgs(in,allowVarArg));
				break;
			case '.':// Index
			{
				if constexpr (FOR_EXPR)
				{
					if (in.peekAt(1) == '.') //is concat (..)
						goto exit;
				}

				in.skip();//skip dot

				SubVarType::NAME res{};
				res.idx = readName(in);

				varDataNeedsSubThing = false;
				// Move auto-clears funcCallData
				varData.back().sub.emplace_back(std::move(funcCallData),std::move(res));
				funcCallData.clear();
				break;
			}
			case '[':// Arr-index
			{
				const char secondCh = in.peekAt(1);

				if (secondCh == '[' || secondCh == '=')//is multi-line string?
				{
					if constexpr (in.settings() & spacedFuncCallStrForm)
					{
						if (!skipped)
							throwSpaceMissingBeforeString(in);
					}
					funcCallData.emplace_back("", readArgs(in,allowVarArg));
					break;
				}
				SubVarType::EXPR<In> res{};

				in.skip();//skip first char
				res.idx = readExpr(in,allowVarArg);
				requireToken(in, "]");

				varDataNeedsSubThing = false;
				// Move auto-clears funcCallData
				varData.back().sub.emplace_back(std::move(funcCallData),std::move(res));
				funcCallData.clear();
				break;
			}
			}
		}

	exit:

		return returnPrefixExprVar<T, FOR_EXPR>(in, varData, funcCallData, varDataNeedsSubThing, opType);
	}

	template<AnyInput In>
	inline Expression<In> readExprParens(In& in, const bool allowVarArg, const bool readBiOp = true) {
		if constexpr (in.settings() & sluaSyn)
		{
			requireToken(in, "(");
			Expression<In> ex = readExpr(in, allowVarArg, readBiOp);
			requireToken(in, ")");
			return ex;
		}
		else
			return readExpr(in, allowVarArg, readBiOp);
	}

	template<AnyInput In>
	inline ExpList<In> readExpList(In& in, const bool allowVarArg)
	{
		/*
			explist ::= exp {‘,’ exp}
		*/
		ExpList<In> ret{};
		ret.emplace_back(readExpr(in, allowVarArg));

		while (checkReadToken(in, ","))
		{
			ret.emplace_back(readExpr(in, allowVarArg));
		}
		return ret;
	}
}