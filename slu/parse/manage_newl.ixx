module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
export module slu.parse.manage_newl;
import slu.parse.input;

namespace slu::parse
{
	export enum class ParseNewlineState : uint8_t
	{
		NONE,
		CARI,
	};

	//Returns if newline was added
	export template <bool skipPreNl>
	bool manageNewlineState(const char ch, ParseNewlineState& nlState, AnyInput auto& in)
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