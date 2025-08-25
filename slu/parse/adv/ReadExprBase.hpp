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
import slu.ast.state;
import slu.parse.input;
#include <slu/parse/adv/SkipSpace.hpp>
#include <slu/parse/adv/RequireToken.hpp>
#include <slu/parse/adv/ReadStringLiteral.hpp>
#include <slu/parse/adv/ReadNumeral.hpp>
#include <slu/parse/basic/ReadOperators.hpp>

namespace slu::parse
{
	template<AnyInput In>
	inline std::pair<lang::ModPath,bool> readModPath(In& in,const std::string& start)
	{
		lang::ModPath mp = { start };
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
	inline lang::ModPath readModPath(In& in) {
		return readModPath(in, readName<NameCatagory::MP_START>(in)).first;
	}
	//Unlike readModPath, doesnt have the ability to do things like `self::xyz`
	template<AnyInput In>
	inline SubModPath readSubModPath(In& in) {
		return readModPath(in,readName(in)).first;
	}

	//Returns if skipped after
	template<AnyInput In>
	inline bool parseVarBase(In& in, const bool allowVarArg, const char firstChar, ExprData<In>& varDataOut, bool& varDataNeedsSubThing)
	{
		if (firstChar == '(')
		{// Must be '(' exp ')'
			in.skip();
			varDataOut = ExprType::Parens<In>(std::make_unique<Expr>(readExpr(in, allowVarArg)));
			varDataNeedsSubThing = true;
			requireToken(in, ")");
			return false;
		}
		if (firstChar == ':')
		{
			requireToken(in, ":>");//Modpath root
			skipSpace(in);
			if (checkToken(in, "::") && in.peekAt(2) != ':') //is '::', but not ':::'
			{
				in.skip(2);
				skipSpace(in);
				varDataOut = ExprType::Global<In>(in.genData.resolveRootName(readModPath(in)));
				return false;
			}
			else
			{
				varDataOut = ExprType::MpRoot{};
				return true;
			}
		}
		// Must be Name, ... or mod path

		std::string start = readName<NameCatagory::MP_START>(in);

		auto [mp, skipped] = readModPath(in, std::move(start));

		if (mp.size() == 1)
		{
			ezmatch(in.genData.resolveNameOrLocal(mp[0]))(
				varcase(const parse::LocalId) {
				varDataOut = ExprType::Local(var);
			},
				varcase(const lang::MpItmId) {
				varDataOut = ExprType::Global<In>(var);
			}
				);
		}
		else
			varDataOut = ExprType::Global<In>(in.genData.resolveName(mp));

		return skipped;
	}

	template<class T,bool FOR_EXPR, AnyInput In>
	inline T returnPrefixExprVar(In& in, std::vector<ExprData<In>>& varData, const bool endsWithArgs,const bool varDataNeedsSubThing,const char opTypeCh)
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
		if constexpr (!FOR_EXPR)
		{
			if (!endsWithArgs)
			{
				if (varDataNeedsSubThing)
					throwRawExpr(in);

				throw UnexpectedCharacterError(std::format(
					"Expected assignment or " LC_function " call, found "
					LUACC_SINGLE_STRING("{}")
					"{}"
					, opType, errorLocStr(in)));
			}
			if (std::holds_alternative<ExprType::Call>(varData.back()))
			{
				ExprType::Call& start = std::get<ExprType::Call>(varData.back());
				StatType::Call res;
				res.args = std::move(start.args);
				res.v = { std::move(*start.v) };

				return std::move(res);
			}
			ExprType::SelfCall& start = std::get<ExprType::SelfCall>(varData.back());
			StatType::SelfCall res;
			res.args = std::move(start.args);
			res.method = start.method;
			res.v = { std::move(*start.v) };

			return std::move(res);
		}
		else
			return std::move(varData.back());
	}
	template<class T,bool boxed, AnyInput In,class... Ts>
	inline T wrapExpr(ast::Position place,ExprData<In>&& expr,Ts&&... extraItems)
	{
		return T{
				parse::ExprUserExpr<boxed>{mayBoxFrom<boxed>(
					Expr{std::move(expr),place}
				)},
				std::move(extraItems)...
		};
	}
	//Doesnt skip space.
	template<class T,bool FOR_EXPR, bool BASIC_ARGS = false, AnyInput In>
	inline T parsePrefixExprVar(In& in, const bool allowVarArg, char firstChar)
	{
		/*
			var ::= baseVar {subvar}

			baseVar ::= Name | ‘(’ exp ‘)’ subvar

			funcArgs ::=  [‘:’ Name] args
			subvar ::= {funcArgs} ‘[’ exp ‘]’ | {funcArgs} ‘.’ Name
		*/

		std::vector<ExprData<In>> varData;
		ast::Position varPlace = in.getLoc();
		bool endsWithArgs = false;
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
				return returnPrefixExprVar<T,FOR_EXPR>(in,varData, endsWithArgs, varDataNeedsSubThing, 0);

			opType = in.peek();
			switch (opType)
			{
			case ',':// Varlist
				if constexpr (FOR_EXPR)
					goto exit;
				else
				{
					if (endsWithArgs)
						throwFuncCallInVarList(in);
					if (varDataNeedsSubThing)
						throwExprInVarList(in);

					in.skip();//skip comma

					varData.emplace_back();
					skipSpace(in);
					varPlace = in.getLoc();
					skipped = parseVarBase(in,allowVarArg, in.peek(), varData.back(), varDataNeedsSubThing);
					break;
				}
			default:
				goto exit;
			case '=':// Assign
				if constexpr (FOR_EXPR)
					goto exit;
				else
				{
					if (endsWithArgs)
						throwFuncCallAssignment(in);
					if (varDataNeedsSubThing)
						throwExprAssignment(in);

					in.skip();//skip eq
					StatType::Assign<In> res{};
					res.vars = std::move(varData);
					res.exprs = readExprList(in,allowVarArg);
					return res;
				}
			case '{':
				if constexpr (BASIC_ARGS)
					goto exit;
				[[fallthrough]];
			case '"':
			case '\'':
			case '('://Funccall
			{
				varData.back() = wrapExpr<ExprType::Call, true, In>(
					varPlace,
					std::move(varData.back()),
					readArgs(in, allowVarArg)
				);
				endsWithArgs = true;
				break;
			}
			case '.':// Index
			{
				if constexpr (FOR_EXPR)
				{
					if (in.peekAt(1) == '.') //is concat or range (..)
						goto exit;
				}
				if (in.peekAt(1) == '*')
				{
					in.skip(2);

					varData.back() = wrapExpr<ExprType::Deref, true, In>(
						varPlace,
						std::move(varData.back())
					);
					varDataNeedsSubThing = false;
					endsWithArgs = false;
					break;
				}
				in.skip();//skip dot

				lang::PoolString name = in.genData.poolStr(readSluTuplableName(in));

				skipSpace(in);
				if (in)
				{//Self call handling.
					const char ch2 = in.peek();
					if (ch2 == '[')
					{
						const char ch3 = in.peekAt(1);
						if (ch3 == '[' || ch3 == '=')
						{
							varData.back() = wrapExpr<ExprType::SelfCall, true, In>(
								varPlace,
								std::move(varData.back()),
								readArgs(in, allowVarArg),
								name
							);
							endsWithArgs = true;
							break;
						}
					}
					else
					{
						if (ch2 == '\'' || ch2 == '"' || ch2 == '{' || ch2 == '(')
						{
							varData.back() = wrapExpr<ExprType::SelfCall, true, In>(
								varPlace,
								std::move(varData.back()),
								readArgs(in, allowVarArg),
								name
							);
							endsWithArgs = true;
							break;
						}
					}
					
				}

				varData.back() = wrapExpr<ExprType::Field<In>, true, In>(
					varPlace,
					std::move(varData.back()),
					name
				);
				varDataNeedsSubThing = false;
				endsWithArgs = false;

				break;
			}
			case '[':// Arr-index
			{
				const char secondCh = in.peekAt(1);

				if (secondCh == '[' || secondCh == '=')//is multi-line string?
				{
					varData.back() = wrapExpr<ExprType::Call, true,In>(
						varPlace,
						std::move(varData.back()),
						readArgs(in, allowVarArg)
					);
					endsWithArgs = true;
					break;
				}

				in.skip();//skip first char
				Expr idx = readExpr(in,allowVarArg);
				requireToken(in, "]");

				varData.back() = wrapExpr<ExprType::Index, true, In>(
					varPlace,
					std::move(varData.back()),
					mayBoxFrom<true>(std::move(idx))
				);
				varDataNeedsSubThing = false;
				endsWithArgs = false;
				break;
			}
			}
		}

	exit:

		return returnPrefixExprVar<T, FOR_EXPR>(in, varData, endsWithArgs, varDataNeedsSubThing, opType);
	}

	template<AnyInput In>
	inline Expr readBasicExpr(In& in, const bool allowVarArg, const bool readBiOp = true) {
		Expr ex = readExpr<true>(in, allowVarArg, readBiOp);
		return ex;
	}

	template<AnyInput In>
	inline ExprList readExprList(In& in, const bool allowVarArg)
	{
		/*
			explist ::= exp {‘,’ exp}
		*/
		ExprList ret{};
		ret.emplace_back(readExpr(in, allowVarArg));

		while (checkReadToken(in, ","))
		{
			ret.emplace_back(readExpr(in, allowVarArg));
		}
		return ret;
	}
}