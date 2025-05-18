/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/Input.hpp>

namespace slu::parse
{
	enum class ParseNewlineState : uint8_t
	{
		NONE,
		CARI,
	};

	//Returns if newline was added
	template <bool skipPreNl>
	inline bool manageNewlineState(const char ch, ParseNewlineState& nlState, AnyInput auto& in)
	{
		switch (nlState)
		{
		case slu::parse::ParseNewlineState::NONE:
			if (ch == '\n')
			{
				if constexpr (skipPreNl)in.skip();
				in.newLine();
				return true;
			}
			else if (ch == '\r')
				nlState = slu::parse::ParseNewlineState::CARI;
			break;
		case slu::parse::ParseNewlineState::CARI:
			if (ch != '\r')
			{//  \r\n, or \r(normal char)
				if constexpr (skipPreNl)in.skip();
				in.newLine();
				nlState = slu::parse::ParseNewlineState::NONE;
				return true;
			}
			else// \r\r
			{
				if constexpr (skipPreNl)in.skip();
				in.newLine();
				return true;
			}
			break;
		}
		return false;
	}
}