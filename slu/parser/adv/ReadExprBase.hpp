/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/ext/CppMatch.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadStringLiteral.hpp>
#include <slu/parser/adv/ReadNumeral.hpp>
#include <slu/parser/basic/ReadOperators.hpp>
#include <slu/parser/errors/CharErrors.h>

namespace slu::parse
{
	template<AnyInput In>
	inline std::pair<ModPath,bool> readModPath(In& in,const std::string& start)
	{
		ModPath mp = { start };
		bool skipped = skipSpace(in);
		while (checkToken(in, "::"))
		{
			const char afterCcChr = in.peekAt(2);
			if (afterCcChr == ':' || afterCcChr == '*' || afterCcChr == '{')
				break;

			in.skip(2);//skip '::'
			skipSpace(in);
			mp.push_back(readName<NameCatagory::MP>(in));
			skipped = skipSpace(in);
		}
		return { mp, skipped };
	}
	template<AnyInput In>
	inline ModPath readModPath(In& in) {
		return readModPath(in, readName<NameCatagory::MP_START>(in)).first;
	}
	//Unlike readModPath, doesnt have the ability to do things like `self::xyz`
	template<AnyInput In>
	inline SubModPath readSubModPath(In& in) {
		return readModPath(in,readName(in)).first;
	}

	//Returns if skipped after
	template<AnyInput In>
	inline bool parseVarBase(In& in, const bool allowVarArg, const char firstChar, Var<In>& varDataOut, bool& varDataNeedsSubThing)
	{
		if (firstChar == '(')
		{// Must be '(' exp ')'
			in.skip();
			Expression<In> res = readExpr(in,allowVarArg);
			requireToken(in, ")");

			varDataOut.base = BaseVarType::EXPR<In>(std::move(res));
			varDataNeedsSubThing = true;
			return false;
		}
		if constexpr (In::settings() & sluSyn)
		{
			if (firstChar == ':')
			{
				requireToken(in, ":>");//Modpath root
				skipSpace(in);
				if(checkToken(in, "::") && in.peekAt(2)!=':') //is '::', but not ':::'
				{
					in.skip(2);
					skipSpace(in);
					varDataOut.base = BaseVarType::NAME<In>(in.genData.resolveRootName(readModPath(in)));
					return false;
				}
				else
				{
					varDataOut.base = BaseVarType::Root{};
					return true;
				}
			}
		}
		// Must be Name, ... or mod path

		//Lua doesnt reserve mp_start names, so doesnt matter
		std::string start = readName<NameCatagory::MP_START>(in);

		if constexpr (In::settings() & sluSyn)
		{
			auto [mp,skipped] = readModPath(in, std::move(start));

			if(mp.size()==1)
			{
				ezmatch(in.genData.resolveNameOrLocal(mp[0]))(
					varcase(const LocalId) {
					varDataOut.base = BaseVarType::Local(var);
				},
					varcase(const MpItmId<In>) {
					varDataOut.base = BaseVarType::NAME<In>(var);
				}
					);
			}
			else
				varDataOut.base = BaseVarType::NAME<In>(in.genData.resolveName(mp));

			return skipped;
		}
		//Check, cuz 'excess elements in struct initializer' happens in normal lua
		varDataOut.base = BaseVarType::NAME<In>(in.genData.resolveName(start));
		return false;
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
	template<class T,bool FOR_EXPR, bool BASIC_ARGS = false, AnyInput In>
	inline T parsePrefixExprVar(In& in, const bool allowVarArg, char firstChar)
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

		bool skipped = parseVarBase(in, allowVarArg, firstChar, varData.back(), varDataNeedsSubThing);

		bool firstRun = true;
		char opType;

		//This requires manual parsing, and stuff (at every step, complex code)
		while (true)
		{
			if (!firstRun || !skipped)
				skipped = skipSpace(in);
			firstRun = false;

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

					varData.emplace_back();

					skipSpace(in);
					skipped = parseVarBase(in,allowVarArg, in.peek(), varData.back(), varDataNeedsSubThing);
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
					res.exprs = readExprList(in,allowVarArg);
					return res;
				}
			}
			case ':'://Self funccall
			{
				if (in.peekAt(1) == ':' || in.peekAt(1) == '>') //is label / '::' / ':>'
					goto exit;
				in.skip();//skip colon
				std::string name = readName(in);

				const bool skippedAfterName = skipSpace(in);

				if constexpr (in.settings() & spacedFuncCallStrForm)
				{
					if (!skippedAfterName)
						throwSpaceMissingBeforeString(in);
				}

				funcCallData.emplace_back(in.genData.resolveUnknown(name), readArgs(in, allowVarArg));
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
				if constexpr (BASIC_ARGS)
					goto exit;
				[[fallthrough]];
			case '('://Funccall
				funcCallData.emplace_back(in.genData.resolveEmpty(), readArgs(in, allowVarArg));
				break;
			case '.':// Index
			{
				if constexpr (FOR_EXPR)
				{
					if (in.peekAt(1) == '.') //is concat or range (..)
						goto exit;
				}
				if constexpr (In::settings() & sluSyn)
				{
					if (in.peekAt(1) == '*')
					{
						in.skip(2);

						varDataNeedsSubThing = false;
						// Move auto-clears funcCallData
						varData.back().sub.emplace_back(std::move(funcCallData), SubVarType::DEREF{});
						funcCallData.clear();

						break;
					}
				}

				in.skip();//skip dot

				SubVarType::NAME<In> res{};
				res.idx = in.genData.resolveUnknown(readSluTuplableName(in));

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
					funcCallData.emplace_back(in.genData.resolveEmpty(), readArgs(in,allowVarArg));
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
	inline Expression<In> readBasicExpr(In& in, const bool allowVarArg, const bool readBiOp = true) {
		if constexpr (in.settings() & sluSyn)
		{
			Expression<In> ex = readExpr<true>(in, allowVarArg, readBiOp);
			return ex;
		}
		else
			return readExpr(in, allowVarArg, readBiOp);
	}

	template<AnyInput In>
	inline ExprList<In> readExprList(In& in, const bool allowVarArg)
	{
		/*
			explist ::= exp {‘,’ exp}
		*/
		ExprList<In> ret{};
		ret.emplace_back(readExpr(in, allowVarArg));

		while (checkReadToken(in, ","))
		{
			ret.emplace_back(readExpr(in, allowVarArg));
		}
		return ret;
	}
}