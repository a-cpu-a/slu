﻿module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <string>
#include <span>
#include <vector>

#include <slu/Panic.hpp>
#include <slu/ext/CppMatch.hpp>
export module slu.paint.basics;

import slu.char_info;
import slu.ast.state;
import slu.paint.sem_output;
import slu.parse.input;
import slu.parse.com.skip_space;

namespace slu::paint
{
	export bool skipSpace(AnySemOutput auto& se) {
		return parse::skipSpace(se);//TODO: identify TODO's FIXME's WIP's, etc
	}
	export template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true, AnySemOutput Se>
	void paintName(Se& se, const std::string_view name)
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);
		if (!name.empty() && name[0] == '0')//does name contain a hex number?
		{//It may be any number, so maybe not the same as 'name'
			while (se.in)
			{
				const char ch = se.in.peek();
				if (!slu::isValidNameChar(ch))
					break;
				se.template add<tok, overlayTok>(1);
				se.in.skip();
			}
			return;
		}
		else if (name.starts_with('$'))
			return;

		for (size_t i = 0; i < name.size(); i++)
		{
			Slu_assert(se.in.peekAt(i) == name[i]);
		}
		se.template add<tok, overlayTok>(name.size());
		se.in.skip(name.size());
	}
	export template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true, AnySemOutput Se>
	void paintName(Se& se, const lang::MpItmId& f) {
		const std::string_view name = se.in.genData.asSv(f);
		paintName<tok, tok, SKIP_SPACE>(se, name);
	}
	export template<Tok tok = Tok::NAME, bool SKIP_SPACE = true, AnySemOutput Se>
	void paintName(Se& se, const lang::MpItmId& f) {
		paintName<tok, tok, SKIP_SPACE>(se, f);
	}
	export template<Tok tok = Tok::NAME, bool SKIP_SPACE = true, AnySemOutput Se>
	void paintPoolStr(Se& se, const lang::PoolString f) {
		const std::string_view name = se.in.genData.asSv(f);
		paintName<tok, tok, SKIP_SPACE>(se, name);
	}
	export template<bool isLocal,Tok tok = Tok::NAME, bool SKIP_SPACE = true, AnySemOutput Se>
	void paintNameOrLocal(Se& se, const parse::LocalOrName<Se,isLocal>& f) {
		if constexpr(isLocal)
			paintName<tok, SKIP_SPACE>(se, se.resolveLocal(f));
		else
			paintName<tok, SKIP_SPACE>(se, f);
	}
	export template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true, size_t TOK_SIZE>
	void paintKw(AnySemOutput auto& se, const char(&tokChr)[TOK_SIZE])
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);
		for (size_t i = 0; i < TOK_SIZE - 1; i++)
		{
			Slu_assert(se.in.peekAt(i) == tokChr[i]);
		}
		se.template add<tok, overlayTok>(TOK_SIZE - 1);
		se.in.skip(TOK_SIZE - 1);
	}
	export template<Tok tok, bool SKIP_SPACE = true, size_t TOK_SIZE>
	void paintKw(AnySemOutput auto& se, const char(&tokChr)[TOK_SIZE]) {
		paintKw<tok, tok, SKIP_SPACE>(se, tokChr);
	}
	export template<Tok tok, Tok overlayTok, bool SKIP_SPACE = true>
	void paintSv(AnySemOutput auto& se, const std::string_view sv)
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);
		for (size_t i = 0; i < sv.size(); i++)
		{
			Slu_assert(se.in.peekAt(i) == sv[i]);
		}
		se.template add<tok, overlayTok>(sv.size());
		se.in.skip(sv.size());
	}
	export template<Tok tok, bool SKIP_SPACE = true>
	void paintSv(AnySemOutput auto& se, const std::string_view sv) {
		paintSv<tok, tok, SKIP_SPACE>(se, sv);
	}
}