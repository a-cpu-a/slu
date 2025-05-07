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
	template<Tok tok=Tok::NAME, bool SKIP_SPACE = true,AnySemOutput Se>
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
	}
	template<Tok tok,bool SKIP_SPACE = true, size_t TOK_SIZE>
	inline void paintKw(AnySemOutput auto& se, const char(&tokChr)[TOK_SIZE]) {
		paintKw<tok,tok,SKIP_SPACE>(se, tokChr);
	}
	template<AnySemOutput Se>
	inline void paintExpr(Se& se, const parse::Expression<Se>& itm)
	{
		se.move(itm.place);
		//todo: unops
		ezmatch(itm.data)();
		if constexpr (Se::settings() & sluaSyn)
		{
			//todo: post unops
		}
	}
	template<Tok tok,AnySemOutput Se>
	inline void paintMp(Se& se, const lang::ModPath& itm)
	{
		//TODO
	}
	template<AnySemOutput Se>
	inline void paintVar(Se& se, const parse::Var<Se>& itm)
	{
		//todo
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
	inline void paintDoEndBlock(Se& se, const parse::Block<Se>& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			se.template add<Tok::BRACES>();
			paintBlock(se, itm);
			skipSpace(se);
			se.template add<Tok::BRACES>();
		}
		else
		{
			paintKw<Tok::COND_STAT>(se, "do");
			paintBlock(se, itm);
			paintKw<Tok::END_STAT>(se, "end");
		}
	}
	template<AnySemOutput Se>
	inline void paintUseVariant(Se& se, const parse::UseVariant& itm)
	{
		ezmatch(itm)(
			// use xxxx::Xxx;
		varcase(const parse::UseVariantType::IMPORT&) {},

			// use xxxx::Xxx::*;
		varcase(const parse::UseVariantType::EVERYTHING_INSIDE) {
			paintKw<Tok::PUNCTUATION>(se, "::");
			paintKw<Tok::PUNCTUATION>(se, "*");
		},
			// use xxxx::Xxx as yyy;
		varcase(const parse::UseVariantType::AS_NAME&) {
			paintKw<Tok::VAR_STAT>(se, "as");
			paintName<Tok::NAME>(se, var);
		},
			// use xxxx::Xxx::{self,a,b,c};
		varcase(const parse::UseVariantType::LIST_OF_STUFF&) {
			paintKw<Tok::PUNCTUATION>(se, "::");
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
			paintDoEndBlock(se, var.bl);
		},
		varcase(const parse::StatementType::FOR_LOOP_GENERIC<Se>&) {
			paintKw<Tok::COND_STAT>(se, "for");

			paintPatOrNamelist(se, var.varNames);
			paintKw<Tok::IN>(se, "in");

			paintExprOrList(se, var.exprs);

			paintDoEndBlock(se, var.bl);
		},
		varcase(const parse::StatementType::IF_THEN_ELSE<Se>&) {
			paintKw<Tok::COND_STAT>(se, "if");
			paintExpr(se, var.cond);
			bool hadBrace = false;
			if constexpr (Se::settings() & sluaSyn)
			{
				skipSpace(se);
				if (se.in.peek() == '{')
				{
					hadBrace = true;
					paintKw<Tok::BRACES>(se, "{");
				}
			}
			else
				paintKw<Tok::COND_STAT>(se, "then");
			paintBlock(se, var.bl);
			for (const auto& [cond,bl] : var.elseIfs)
			{
				if constexpr (Se::settings() & sluaSyn)
				{
					if(hadBrace)
						paintKw<Tok::BRACES>(se, "}");
					paintKw<Tok::COND_STAT>(se, "else");
					paintKw<Tok::COND_STAT>(se, "if");
					paintExpr(se, cond);
					skipSpace(se);
					if (se.in.peek() == '{')
					{
						hadBrace = true;
						paintKw<Tok::BRACES>(se, "{");
					}
				}
				else
				{
					paintKw<Tok::COND_STAT>(se, "elseif");
					paintExpr(se, cond);
					paintKw<Tok::COND_STAT>(se, "then");
				}
				paintBlock(se, var.bl);
			}
			if (var.elseBlock.has_value())
			{
				if constexpr (Se::settings() & sluaSyn)
				{
					if (hadBrace)
						paintKw<Tok::BRACES>(se, "}");
					paintKw<Tok::COND_STAT>(se, "else");
					skipSpace(se);
					if (se.in.peek() == '{')
					{
						hadBrace = true;
						paintKw<Tok::BRACES>(se, "{");
					}
				}
				else
					paintKw<Tok::COND_STAT>(se, "else");
				paintBlock(se, var.bl);
			}

			if constexpr (Se::settings() & sluaSyn)
			{
				if (hadBrace)
					paintKw<Tok::BRACES>(se, "}");
			}
			else
				paintKw<Tok::END_STAT>(se, "end");

		},
		varcase(const parse::StatementType::ASSIGN<Se>&) {
			paintVarList(se, var.vars);
			skipSpace(se);
			paintKw<Tok::ASSIGN>(se, "=");
			paintExprList(se, var.exprs);
		},
		varcase(const parse::StatementType::LOCAL_ASSIGN<Se>&) {
			paintKw<Tok::VAR_STAT>(se, "local");

			paintPatOrNamelist(se, var.names);
			
			skipSpace(se);
			se.template add<Tok::ASSIGN>();
			paintExprList(se, var.exprs);
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

			skipSpace(se);

			se.template add<Tok::BRACES>();
			paintBlock(se, var.bl);
			skipSpace(se);
			se.template add<Tok::BRACES>();
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