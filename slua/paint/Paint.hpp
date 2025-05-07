/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/Settings.hpp>
#include <slua/ext/CppMatch.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/State.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/VecInput.hpp>
#include <slua/paint/SemOutputStream.hpp>

namespace slua::paint
{
	using parse::sluaSyn;

	inline bool skipSpace(AnySemOutput auto& se) {
		//TODO: special comment coloring
		return parse::skipSpace(se.in);//Doesnt write to the sem out thing
	}
	template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true,AnySemOutput Se>
	inline void paintName(Se& se, const parse::MpItmId<Se>& f)
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);
		const std::string_view name = se.in.genData.asSv(f);
		for (size_t i = 0; i < name.size(); i++)
		{
			_ASSERT(se.in.peekAt(i) == name[i]);
		}
		se.template add<tok>(name.size());
		se.in.skip(name.size());
	}
	template<Tok tok = Tok::NAME, bool SKIP_SPACE = true, AnySemOutput Se>
	inline void paintName(Se& se, const parse::MpItmId<Se>& f) {
		paintName<tok, tok, SKIP_SPACE>(se, f);
	}
	template<Tok tok, Tok overlayTok,bool SKIP_SPACE = true, size_t TOK_SIZE>
	inline void paintKw(AnySemOutput auto& se, const char(&tokChr)[TOK_SIZE])
	{
		if constexpr(SKIP_SPACE)
			skipSpace(se);
		for (size_t i = 0; i < TOK_SIZE-1; i++)
		{
			_ASSERT(se.in.peekAt(i) == tokChr[i]);
		}
		se.template add<tok, overlayTok>(TOK_SIZE - 1);
		se.in.skip(TOK_SIZE - 1);
	}
	template<Tok tok,bool SKIP_SPACE = true, size_t TOK_SIZE>
	inline void paintKw(AnySemOutput auto& se, const char(&tokChr)[TOK_SIZE]) {
		paintKw<tok,tok,SKIP_SPACE>(se, tokChr);
	}
	template<Tok tok, Tok overlayTok,bool SKIP_SPACE = true>
	inline void paintSv(AnySemOutput auto& se, const std::string_view sv)
	{
		if constexpr(SKIP_SPACE)
			skipSpace(se);
		for (size_t i = 0; i < sv.size(); i++)
		{
			_ASSERT(se.in.peekAt(i) == tokChr[i]);
		}
		se.template add<tok, overlayTok>(sv.size());
		se.in.skip(sv.size());
	}
	template<Tok tok, bool SKIP_SPACE = true>
	inline void paintSv(AnySemOutput auto& se, const std::string_view sv) {
		paintSv<tok, tok, SKIP_SPACE>(se, sv);
	}
	template<bool SKIP_SPACE = true>
	inline void paintString(AnySemOutput auto& se, const std::string_view sv,const Position end,const Tok tint) 
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);

		se.template add<Tok::STRING_OUT>(tint);
		const char fChar = se.in.get();
		if (fChar == '[')
		{
			size_t level = 0;
			while (se.in.peek() == '=')
			{
				level++;
				se.template add<Tok::STRING_OUT>(tint);
				se.in.skip();
			}
			se.template add<Tok::STRING_OUT>(tint);
			se.in.skip();

			se.template move<Tok::STRING>();
			se.template replPrev<Tok::STRING_OUT>(tint,level+2);
		}
		else
		{
			se.template move<Tok::STRING>();
			se.template replPrev<Tok::STRING_OUT>(tint);
		}
	}
	template<bool SKIP_SPACE = true>
	inline void paintNumber(AnySemOutput auto& se, const Tok tint) 
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);
	}
	template<AnySemOutput Se>
	inline void paintExpr(Se& se, const parse::Expression<Se>& itm,const Tok tint = Tok::NONE)
	{
		se.move(itm.place);
		if (std::holds_alternative<parse::ExprType::MULTI_OPERATION>(itm.data))
		{
			//complex version
		}
		//todo: unops
		ezmatch(itm.data)(
		varcase(const parse::ExprType::MULTI_OPERATION<Se>&) {
			Slua_panic("parse::ExprType::MULTI_OPERATION should have been handled outside!");
		},
		varcase(const parse::ExprType::FALSE) {
			paintKw<Tok::BUILITIN_VAR>(se, "false");
		},
		varcase(const parse::ExprType::TRUE) {
			paintKw<Tok::BUILITIN_VAR>(se, "true");
		},
		varcase(const parse::ExprType::NIL) {
			paintKw<Tok::BUILITIN_VAR>(se, "nil");
		},
		varcase(const parse::ExprType::VARARGS) {
			paintKw<Tok::PUNCTUATION>(se, "...");
		},
		varcase(const parse::ExprType::OPEN_RANGE) {
			paintKw<Tok::RANGE>(se, "...");
		},
		varcase(const parse::ExprType::LITERAL_STRING&) {
			paintString(se, var.v,var.end,tint);
		},
		varcase(const parse::ExprType::NUMERAL) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::NUMERAL_I128) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::NUMERAL_U128) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::NUMERAL_I64) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::NUMERAL_U64) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::TYPE_EXPR&) {
			paintTypeExpr(se, var);
		},
		varcase(const parse::ExprType::TRAIT_EXPR&) {
			paintTraitExpr(se, var);
		},
		varcase(const parse::ExprType::ARRAY_CONSTRUCTOR<Se>&) {
			paintKw<Tok::GEN_OP>(se, "[");
			paintExpr(se, *var.val);
			paintKw<Tok::PUNCTUATION>(se, ";");
			paintExpr(se, *var.size);
			paintKw<Tok::GEN_OP>(se, "]");
		},
		varcase(const parse::ExprType::FUNCTION_DEF<Se>&) {
			paintFuncDef(se, var.v,parse::MpItmId<Se>(SIZE_MAX));
		},
		varcase(const parse::ExprType::FUNC_CALL<Se>&) {
			paintLimPrefixExpr(se, *var.val);
			paintArgChain(se, var.argChain);
		}
		);
		if constexpr (Se::settings() & sluaSyn)
		{
			//todo: post unops
		}
	}
	template<Tok tok,AnySemOutput Se>
	inline void paintMp(Se& se, const parse::MpItmId<Se>& itm)
	{
		const lang::ViewModPath mp = itm.asVmp(se.in.genData);
		for (auto& i : mp)
		{
			if(i=="self")
				paintKw<Tok::VAR_STAT>(se, "self");
			else if(i=="Self" || i=="crate")
				paintSv<Tok::CON_STAT>(se, i);
			else
				paintSv<tok>(se, i);

			if (&i != &mp.back())
				paintKw<Tok::MP>(se, "::");
		}
	}
	template<AnySemOutput Se>
	inline void paintVar(Se& se, const parse::Var<Se>& itm)
	{
		ezmatch(itm.base)(
		varcase(const parse::BaseVarType::NAME<Se>&) {
			if constexpr (Se::settings() & sluaSyn)
			{
				if (var.hasDeref)
				{
					paintKw<Tok::DEREF>(se, "*");
					paintName<Tok::NAME, Tok::DEREF>(se, var.v);
					return;//From match, not func
				}
			}
			paintName<Tok::NAME>(se, var.v);
		},
		varcase(const parse::BaseVarType::EXPR<Se>&) {
			paintKw<Tok::GEN_OP>(se, "(");
			paintExpr(se, var.start);
			paintKw<Tok::GEN_OP>(se, ")");
		},
		varcase(const parse::BaseVarType::EXPR_DEREF_NO_SUB<Se>&) {
			paintKw<Tok::DEREF>(se, "*");
			paintKw<Tok::GEN_OP, Tok::DEREF>(se, "(");
			paintExpr(se, var.start);
			paintKw<Tok::GEN_OP, Tok::DEREF>(se, ")");
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintPat(Se& se, const parse::Pat& itm)
	{

	}
	template<Tok tok, AnySemOutput Se>
	inline void paintNameList(Se& se, const std::vector<parse::MpItmId<Se>>& itm)
	{
		for (const parse::MpItmId<Se>& i : itm)
		{
			paintName<tok>(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<AnySemOutput Se>
	inline void paintPatOrNamelist(Se& se, const auto& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
			paintPat(se, itm);
		else
		{
			if constexpr (std::same_as<decltype(itm), parse::MpItmId<Se>>)
				paintName<Tok::NAME>(se, itm);
			else
				paintNameList<Tok::NAME>(se, itm);
		}
	}
	template<AnySemOutput Se>
	inline void paintVarList(Se& se, const std::vector<parse::Var<Se>>& itm)
	{
		for (const parse::Var<Se>& i : itm)
		{
			paintVar(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<AnySemOutput Se>
	inline void paintLimPrefixExpr(Se& se, const parse::LimPrefixExpr<Se>& itm)
	{
		//TODO
	}
	template<AnySemOutput Se>
	inline void paintArgChain(Se& se, const std::vector<parse::ArgFuncCall<Se>>& itm)
	{
		//TODO
	}
	template<AnySemOutput Se>
	inline void paintEndBlock(Se& se, const parse::Block<Se>& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			paintBlock(se, itm);
			skipSpace(se);
			paintKw<Tok::BRACES>(se, "}");
		}
		else
		{
			paintBlock(se, itm);
			paintKw<Tok::END_STAT>(se, "end");
		}
	}
	template<AnySemOutput Se>
	inline void paintDoEndBlock(Se& se, const parse::Block<Se>& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
			paintKw<Tok::BRACES>(se, "{");
		else
			paintKw<Tok::COND_STAT>(se, "do");
		paintEndBlock(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintTraitExpr(Se& se, const parse::TraitExpr& itm)
	{
	}
	template<AnySemOutput Se>
	inline void paintTypeExpr(Se& se, const parse::TypeExpr& itm)
	{
	}
	template<AnySemOutput Se>
	inline void paintSafety(Se& se, const parse::OptSafety itm)
	{
		switch (itm)
		{
		case parse::OptSafety::SAFE:
			paintKw<Tok::FN_STAT>(se, "safe");
			break;
		case parse::OptSafety::UNSAFE:
			paintKw<Tok::FN_STAT>(se, "unsafe");
			break;
		case parse::OptSafety::DEFAULT:
		default:
			break;
		}
	}
	template<bool DO_END,AnySemOutput Se>
	inline void paintStatOrRet(Se& se, const parse::Block<Se>& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			skipSpace(se);
			bool hadBrace = false;
			if (se.in.peek() == '{')
			{
				hadBrace = true;
				paintKw<Tok::BRACES>(se, "{");
			}

			paintBlock(se, itm);

			if(hadBrace)
				paintKw<Tok::BRACES>(se, "}");
			return;
		}
		if constexpr(DO_END)
			paintDoEndBlock(se, itm);
		else
			paintBlock(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintFuncDef(Se& se, const parse::Function<Se>& func,const parse::MpItmId<Se> name)
	{
		//TODO:
		//if (itm.func.exported)
		//	paintKw<Tok::CON_STAT, Tok::EX_TINT>(se, "ex");

		if constexpr (Se::settings() & sluaSyn)
			paintSafety(se, func.safety);
		paintKw<Tok::FN_STAT>(se, "function");
		paintName<Tok::NAME>(se, name);
		paintKw<Tok::GEN_OP>(se, "(");
		for (const parse::Parameter<Se>& i : func.params)
		{
			paintPatOrNamelist(se, i.name);

			if (&i != &func.params.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
		paintKw<Tok::GEN_OP>(se, ")");
		if constexpr (Se::settings() & sluaSyn)
		{
			if (func.retType.has_value())
			{
				paintKw<Tok::GEN_OP>(se, "->");
				paintTypeExpr(se, *func.retType);
			}
			paintKw<Tok::BRACES>(se, "{");
		}
		paintEndBlock(se, func.block);
		
	}
	template<AnySemOutput Se>
	inline void paintUseVariant(Se& se, const parse::UseVariant& itm)
	{
		ezmatch(itm)(
			// use xxxx::Xxx;
		varcase(const parse::UseVariantType::IMPORT&) {},

			// use xxxx::Xxx::*;
		varcase(const parse::UseVariantType::EVERYTHING_INSIDE) {
			paintKw<Tok::MP>(se, "::");
			paintKw<Tok::PUNCTUATION>(se, "*");
		},
			// use xxxx::Xxx as yyy;
		varcase(const parse::UseVariantType::AS_NAME&) {
			paintKw<Tok::VAR_STAT>(se, "as");
			paintName<Tok::NAME>(se, var);
		},
			// use xxxx::Xxx::{self,a,b,c};
		varcase(const parse::UseVariantType::LIST_OF_STUFF&) {
			paintKw<Tok::MP>(se, "::");
			paintKw<Tok::BRACES>(se, "{");
			for (const parse::MpItmId<Se>& i : var)
			{
				paintName<Tok::NAME>(se, i);

				if (&i != &var.back())
					paintKw<Tok::PUNCTUATION>(se, ",");
			}
			paintKw<Tok::BRACES>(se, "}");
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintStat(Se& se, const parse::Statement<Se>& itm)
	{
		se.move(itm.place);
		ezmatch(itm.data)(
		varcase(const parse::StatementType::BLOCK<Se>&) {
			paintDoEndBlock(se, var.bl);
		},
		varcase(const parse::StatementType::FOR_LOOP_NUMERIC<Se>&) {
			paintKw<Tok::COND_STAT>(se, "for");

			paintPatOrNamelist(se, var.varName);
			paintKw<Tok::ASSIGN>(se, "=");

			paintExpr(se, var.start);
			paintKw<Tok::PUNCTUATION>(se, ",");
			paintExpr(se, var.end);
			if (var.step.has_value())
			{
				paintKw<Tok::PUNCTUATION>(se, ",");
				paintExpr(se, *var.step);
			}
			paintStatOrRet<true>(se, var.bl);
		},
		varcase(const parse::StatementType::FOR_LOOP_GENERIC<Se>&) {
			paintKw<Tok::COND_STAT>(se, "for");

			paintPatOrNamelist(se, var.varNames);
			paintKw<Tok::IN>(se, "in");

			paintExprOrList(se, var.exprs);

			paintStatOrRet<true>(se, var.bl);
		},
		varcase(const parse::StatementType::WHILE_LOOP<Se>&) {
			paintKw<Tok::COND_STAT>(se, "while");

			paintExpr(se, var.cond);

			paintStatOrRet<true>(se, var.bl);
		},
		varcase(const parse::StatementType::REPEAT_UNTIL<Se>&) {
			paintKw<Tok::COND_STAT>(se, "repeat");

			paintStatOrRet<false>(se, var.bl);

			paintKw<Tok::COND_STAT>(se, "until");

			paintExpr(se, var.cond);
		},
		varcase(const parse::StatementType::IF_THEN_ELSE<Se>&) {
			paintKw<Tok::COND_STAT>(se, "if");
			paintExpr(se, var.cond);

			if constexpr (!(Se::settings() & sluaSyn))
				paintKw<Tok::COND_STAT>(se, "then");

			paintStatOrRet<false>(se, var.bl);
			for (const auto& [cond,bl] : var.elseIfs)
			{
				if constexpr (Se::settings() & sluaSyn)
				{
					paintKw<Tok::COND_STAT>(se, "else");
					paintKw<Tok::COND_STAT>(se, "if");
					paintExpr(se, cond);
				}
				else
				{
					paintKw<Tok::COND_STAT>(se, "elseif");
					paintExpr(se, cond);
					paintKw<Tok::COND_STAT>(se, "then");
				}
				paintStatOrRet<false>(se, bl);
			}
			if (var.elseBlock.has_value())
			{
				paintKw<Tok::COND_STAT>(se, "else");
				paintStatOrRet<false>(se, *var.elseBlock);
			}

			if constexpr (!(Se::settings() & sluaSyn))
				paintKw<Tok::END_STAT>(se, "end");

		},
		varcase(const parse::StatementType::FUNC_CALL<Se>&) {
			paintLimPrefixExpr(se, *var.val);
			paintArgChain(se, var.argChain);
		},
		varcase(const parse::StatementType::ASSIGN<Se>&) {
			paintVarList(se, var.vars);
			paintKw<Tok::ASSIGN>(se, "=");
			paintExprList(se, var.exprs);
		},
		varcase(const parse::StatementType::LOCAL_ASSIGN<Se>&) {
			paintKw<Tok::VAR_STAT>(se, "local");

			paintPatOrNamelist(se, var.names);
			
			paintKw<Tok::ASSIGN>(se, "=");
			paintExprList(se, var.exprs);
		},
		varcase(const parse::StatementType::FUNCTION_DEF<Se>&) {
			se.move(var.place);//TODO: parse this correctly
			paintFuncDef(se, var.func, var.name);
		},
		varcase(const parse::StatementType::LOCAL_FUNCTION_DEF<Se>&) {
			se.move(var.place);//TODO: parse this correctly
			paintKw<Tok::FN_STAT>(se, "local");
			paintFuncDef(se, var.func, var.name);
		},
		varcase(const parse::StatementType::DROP<Se>&) {
			paintKw<Tok::DROP_STAT>(se, "drop");
			paintExpr(se, var.expr);
		},
		varcase(const parse::StatementType::MOD_CRATE) {
			paintKw<Tok::CON_STAT>(se, "mod");
			paintKw<Tok::CON_STAT>(se, "crate");
		},
		varcase(const parse::StatementType::MOD_SELF) {
			paintKw<Tok::CON_STAT>(se, "mod");
			paintKw<Tok::VAR_STAT>(se, "self");
		},
		varcase(const parse::StatementType::BREAK) {
			paintKw<Tok::COND_STAT>(se, "break");
		},
		varcase(const parse::StatementType::SEMICOLON) {
			paintKw<Tok::PUNCTUATION>(se, ";");
		},
		varcase(const parse::StatementType::UNSAFE_LABEL) {
			paintKw<Tok::PUNCTUATION>(se, ":::");
			paintKw<Tok::FN_STAT>(se, "unsafe");
			paintKw<Tok::PUNCTUATION>(se, ":");
		},
		varcase(const parse::StatementType::SAFE_LABEL) {
			paintKw<Tok::PUNCTUATION>(se, ":::");
			paintKw<Tok::FN_STAT>(se, "safe");
			paintKw<Tok::PUNCTUATION>(se, ":");
		},
		varcase(const parse::StatementType::LABEL<Se>&) {
			paintKw<Tok::PUNCTUATION>(se, parse::sel<Se>("::", ":::"));
			paintName<Tok::NAME_LABEL>(se, var.v);
			paintKw<Tok::PUNCTUATION>(se, parse::sel<Se>("::", ":"));
		},
		varcase(const parse::StatementType::GOTO<Se>&) {
			paintKw<Tok::COND_STAT>(se, "goto");
			paintName<Tok::NAME_LABEL>(se, var.v);
		},
		varcase(const parse::StatementType::MOD_DEF<Se>&) {
			if (var.exported)
				paintKw<Tok::CON_STAT, Tok::EX_TINT>(se, "ex");
			paintKw<Tok::CON_STAT>(se, "mod");
			paintName<Tok::NAME>(se, var.name);
		},
		varcase(const parse::StatementType::USE&) {
			if constexpr (Se::settings() & sluaSyn)
			{
				if (var.exported)
					paintKw<Tok::VAR_STAT, Tok::EX_TINT>(se, "ex");

				paintKw<Tok::CON_STAT>(se, "use");
				paintMp<Tok::NAME>(se, var.base);
				paintUseVariant(se, var.useVariant);
			}
		},
		varcase(const parse::StatementType::MOD_DEF_INLINE<Se>&) {
			if (var.exported)
				paintKw<Tok::CON_STAT, Tok::EX_TINT>(se, "ex");
			paintKw<Tok::CON_STAT>(se, "mod");
			paintName<Tok::NAME>(se, var.name);

			paintKw<Tok::CON_STAT>(se, "as");

			paintKw<Tok::BRACES>(se, "{");
			paintBlock(se, var.bl);
			paintKw<Tok::BRACES>(se, "}");
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintExprList(Se& se, const parse::ExpList<Se>& itm)
	{
		for (const parse::Expression<Se>& i : itm)
		{
			paintExpr(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<AnySemOutput Se>
	inline void paintExprOrList(Se& se, const parse::ExpList<Se>& itm) {
		return paintExprList(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintExprOrList(Se& se, const parse::Expression<Se>& itm) {
		return paintExpr(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintBlock(Se& se, const parse::Block<Se>& itm)
	{
		se.move(itm.start);
		for (const parse::Statement<Se>& stat : itm.statList)
		{
			paintStat(se, stat);
		}
		if (itm.hadReturn)
		{
			if constexpr(Se::settings()&sluaSyn)
			{
				skipSpace(se);
				if (se.in.peek() == 'd')
					paintKw<Tok::FN_STAT, false>(se, "do");
			}
			paintKw<Tok::FN_STAT>(se,"return");
			paintExprList(se, itm.retExprs);
			//for ;
			se.template move<Tok::PUNCTUATION>(itm.end);
		}
	}
	template<AnySemOutput Se>
	inline void paintFile(Se& se, const parse::ParsedFile<Se>& f)
	{
		paintBlock(se, f.code);
	}
}