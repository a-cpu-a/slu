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
#include <slua/parser/basic/CharInfo.hpp>
#include <slua/paint/SemOutputStream.hpp>
#include <slua/paint/PaintOps.hpp>

namespace slua::paint
{
	using parse::sluaSyn;

	inline bool skipSpace(AnySemOutput auto& se) {
		//TODO: special comment coloring
		return parse::skipSpace(se.in);//Doesnt write to the sem out thing
	}
	template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true, AnySemOutput Se>
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