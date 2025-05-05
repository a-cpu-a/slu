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
		//todo
		//se.in.genData;
		
	}
	template<Tok tok,bool SKIP_SPACE = true, size_t TOK_SIZE>
	inline void paintKw(AnySemOutput auto& se, const char(&tokChr)[TOK_SIZE])
	{
		if constexpr(SKIP_SPACE)
			skipSpace(se);
		for (size_t i = 0; i < TOK_SIZE-1; i++)
		{
			_ASSERT(se.in.peekAt(i) == tokChr[i]);
		}
		se.template add<tok>(TOK_SIZE - 1);
	}
	template<AnySemOutput Se>
	inline void paintExpr(Se& se, const parse::Expression<Se>& f)
	{
		se.move(f.place);
		//todo: unops
		ezmatch(f.data)();
		if constexpr (Se::settings() & sluaSyn)
		{
			//todo: post unops
		}
	}
	template<AnySemOutput Se>
	inline void paintVarList(Se& se, const std::vector<parse::Var<Se>>& f)
	{
		//todo
	}
	template<AnySemOutput Se>
	inline void paintStat(Se& se, const parse::Statement<Se>& f)
	{
		se.move(f.place);
		ezmatch(f.data)(
		varcase(const parse::StatementType::BLOCK<Se>&) {
			se.template add<Tok::BRACES>();
			paintBlock(se, var.bl);
			skipSpace(se);
			se.template add<Tok::BRACES>();
		},
		varcase(const parse::StatementType::ASSIGN<Se>&) {
			paintVarList(se, var.vars);
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
		varcase(const parse::StatementType::GOTO<Se>&) {
			paintKw<Tok::COND_STAT>(se, "goto");
			paintName<Tok::NAME_LABEL>(se, var.v);
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintExprList(Se& se, const parse::ExpList<Se>& f)
	{
		for (const parse::Expression<Se>& i : f)
		{
			paintExpr(se, i);
			skipSpace(se);
			if (&i != &f.back())
				se.template add<Tok::PUNCTUATION>();
		}
	}
	template<AnySemOutput Se>
	inline void paintBlock(Se& se, const parse::Block<Se>& f)
	{
		se.move(f.start);
		for (const parse::Statement<Se>& stat : f.statList)
		{
			paintStat(se, stat);
		}
		if (f.hadReturn)
		{
			if constexpr(Se::settings()&sluaSyn)
			{
				skipSpace(se);
				if (se.in.peek() == 'd')
					paintKw<Tok::FN_STAT, false>(se, "do");
			}
			paintKw<Tok::FN_STAT>(se,"return");
			paintExprList(se,f.retExprs);
			//for ;
			se.template move<Tok::PUNCTUATION>(f.end);
		}
	}
	template<AnySemOutput Se>
	inline void paintFile(Se& se, const parse::ParsedFile<Se>& f)
	{
		paintBlock(se, f.code);
	}
}