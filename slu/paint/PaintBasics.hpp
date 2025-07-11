﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/Settings.hpp>
#include <slu/ext/CppMatch.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/VecInput.hpp>
#include <slu/parser/basic/CharInfo.hpp>
#include <slu/paint/SemOutputStream.hpp>
#include <slu/paint/PaintOps.hpp>

namespace slu::paint
{
	using parse::sluSyn;

	inline bool skipSpace(AnySemOutput auto& se) {
		return parse::skipSpace(se);//TODO: identify TODO's FIXME's WIP's, etc
	}
	template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true, AnySemOutput Se>
	inline void paintName(Se& se, const parse::MpItmId<Se>& f)
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);
		const std::string_view name = se.in.genData.asSv(f);
		if (!name.empty() && name[0] == '0')//does name contain a hex number?
		{//It may be any number, so maybe not the same as 'name'
			while (se.in)
			{
				const char ch = se.in.peek();
				if (!parse::isValidNameChar(ch))
					break;
				se.template add<tok, overlayTok>(1);
				se.in.skip();
			}
		}
		else
		{
			for (size_t i = 0; i < name.size(); i++)
			{
				_ASSERT(se.in.peekAt(i) == name[i]);
			}
			se.template add<tok, overlayTok>(name.size());
			se.in.skip(name.size());
		}
	}
	template<Tok tok = Tok::NAME, bool SKIP_SPACE = true, AnySemOutput Se>
	inline void paintName(Se& se, const parse::MpItmId<Se>& f) {
		paintName<tok, tok, SKIP_SPACE>(se, f);
	}
	template<bool isLocal,Tok tok = Tok::NAME, bool SKIP_SPACE = true, AnySemOutput Se>
	inline void paintNameOrLocal(Se& se, const parse::LocalOrName<Se,isLocal>& f) {
		if constexpr(isLocal)
			paintName<tok, SKIP_SPACE>(se, se.resolveLocal(f));
		else
			paintName<tok, SKIP_SPACE>(se, f);
	}
	template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true, size_t TOK_SIZE>
	inline void paintKw(AnySemOutput auto& se, const char(&tokChr)[TOK_SIZE])
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);
		for (size_t i = 0; i < TOK_SIZE - 1; i++)
		{
			_ASSERT(se.in.peekAt(i) == tokChr[i]);
		}
		se.template add<tok, overlayTok>(TOK_SIZE - 1);
		se.in.skip(TOK_SIZE - 1);
	}
	template<Tok tok, bool SKIP_SPACE = true, size_t TOK_SIZE>
	inline void paintKw(AnySemOutput auto& se, const char(&tokChr)[TOK_SIZE]) {
		paintKw<tok, tok, SKIP_SPACE>(se, tokChr);
	}
	template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true>
	inline void paintSv(AnySemOutput auto& se, const std::string_view sv)
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);
		for (size_t i = 0; i < sv.size(); i++)
		{
			_ASSERT(se.in.peekAt(i) == sv[i]);
		}
		se.template add<tok, overlayTok>(sv.size());
		se.in.skip(sv.size());
	}
	template<Tok tok, bool SKIP_SPACE = true>
	inline void paintSv(AnySemOutput auto& se, const std::string_view sv) {
		paintSv<tok, tok, SKIP_SPACE>(se, sv);
	}
}