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
#include <slua/parser/Input.hpp>
#include <slua/parser/State.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/semtok/SemOutputStream.hpp>

namespace slua::stok
{
	inline bool skipSpace(AnySemOutput auto& se) {
		return parse::skipSpace(se.in);
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
	inline void paintStat(Se& se, const parse::Statement<Se>& f)
	{
		se.move(f.place);
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
			//TODO:
			skipSpace(se);
			if (se.in.peek() == 'd');
				paintKw<Tok::RET_STAT,false>(se, "do");
			paintKw<Tok::RET_STAT>(se,"return");
		}
	}
	template<AnySemOutput Se>
	inline void paintFile(Se& se, const parse::ParsedFile<Se>& f)
	{
		paintBlock(se, f.code);
	}
}